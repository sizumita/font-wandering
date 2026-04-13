mod text_input;

use std::{collections::HashMap, sync::Arc};

use gpui::{
    AppContext, AsyncWindowContext, Bounds, Context, Corners, FontWeight, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point, Render, RenderImage, ScrollDelta,
    ScrollWheelEvent, WeakEntity, Window, canvas, div, fill, img, outline, point, prelude::*, px,
    quad, rgb, size, transparent_black, white,
};
use image::{Frame, ImageBuffer, Rgba};
use smallvec::smallvec;

use crate::{
    cache::AnalysisCache,
    font_pipeline::FontPipeline,
    ml_pipeline::MlPipeline,
    models::{
        AnalysisProgress, AnalysisRequest, AnalysisResult, AnalysisStage, AnalysisStats,
        DEFAULT_INPUT_TEXT, EmbeddingPoint, ExcludedFace, FontFaceMetadata, FontSample, MapPoint,
        RenderSpec,
    },
};

use text_input::TextInput;
pub use text_input::bind_keys;

const PLOT_PADDING: f32 = 28.0;
const THUMBNAIL_MIN_SIDE: f32 = 28.0;
const THUMBNAIL_MAX_SIDE: f32 = 40.0;
const THUMBNAIL_BASE_RATIO: f32 = 0.06;
const THUMBNAIL_CORNER_RADIUS: f32 = 6.0;
const THUMBNAIL_BORDER_WIDTH: f32 = 1.25;
const THUMBNAIL_HOVER_SCALE: f32 = 1.14;
const THUMBNAIL_SELECTED_SCALE: f32 = 1.28;
const THUMBNAIL_RING_GAP: f32 = 3.0;

#[derive(Clone, Debug)]
struct PlotState {
    bounds: Option<Bounds<Pixels>>,
    pan: Point<Pixels>,
    zoom: f32,
    drag_last: Option<Point<Pixels>>,
    drag_moved: bool,
}

impl Default for PlotState {
    fn default() -> Self {
        Self {
            bounds: None,
            pan: point(px(0.0), px(0.0)),
            zoom: 1.0,
            drag_last: None,
            drag_moved: false,
        }
    }
}

#[derive(Clone, Debug)]
struct PlotPoint {
    face_id: fontdb::ID,
    bounds: Bounds<Pixels>,
    cluster_id: i32,
    thumbnail: Arc<RenderImage>,
    selected: bool,
    hovered: bool,
    z_order: u8,
}

#[derive(Clone, Debug)]
struct PlotSourcePoint {
    face_id: fontdb::ID,
    x: f32,
    y: f32,
    cluster_id: i32,
    thumbnail: Arc<RenderImage>,
    selected: bool,
    hovered: bool,
}

pub struct RootView {
    input: gpui::Entity<TextInput>,
    font_pipeline: FontPipeline,
    ml_pipeline: MlPipeline,
    cache: AnalysisCache,
    progress: AnalysisProgress,
    result: Option<AnalysisResult>,
    selected_face: Option<fontdb::ID>,
    hovered_face: Option<fontdb::ID>,
    detail_face: Option<fontdb::ID>,
    plot: PlotState,
    thumbnail_cache: HashMap<fontdb::ID, Arc<RenderImage>>,
    is_running: bool,
    error_message: Option<String>,
}

impl RootView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| TextInput::new(DEFAULT_INPUT_TEXT, "解析したい文字列", cx));

        cx.on_next_frame(window, |this, window, cx| {
            cx.focus_view(&this.input, window);
        });

        Self {
            input,
            font_pipeline: FontPipeline::new(),
            ml_pipeline: MlPipeline::new(),
            cache: AnalysisCache::default(),
            progress: AnalysisProgress::idle(),
            result: None,
            selected_face: None,
            hovered_face: None,
            detail_face: None,
            plot: PlotState::default(),
            thumbnail_cache: HashMap::new(),
            is_running: false,
            error_message: None,
        }
    }

    fn analyze(&mut self, _: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        if self.is_running {
            return;
        }

        let text = self.input.read(cx).content().to_string();
        if text.trim().is_empty() {
            self.fail("入力文字列が空です".to_string(), cx);
            return;
        }

        let request = AnalysisRequest::new(
            text,
            RenderSpec::default(),
            self.ml_pipeline.model_revision(),
        );
        let font_pipeline = self.font_pipeline.clone();
        let ml_pipeline = self.ml_pipeline.clone();
        let cache = self.cache.clone();

        self.is_running = true;
        self.result = None;
        self.selected_face = None;
        self.hovered_face = None;
        self.detail_face = None;
        self.plot = PlotState::default();
        self.thumbnail_cache.clear();
        self.error_message = None;
        self.progress = AnalysisProgress::new(
            AnalysisStage::ScanningFonts,
            0,
            0,
            "フォントカタログを走査しています",
        );
        cx.notify();

        cx.spawn_in(
            window,
            move |view: WeakEntity<RootView>, cx: &mut AsyncWindowContext| {
                let mut cx = cx.clone();
                async move {
                    let faces: Vec<FontFaceMetadata> = cx
                        .background_executor()
                        .spawn({
                            let font_pipeline = font_pipeline.clone();
                            async move { font_pipeline.collect_faces() }
                        })
                        .await;
                    let total_faces = faces.len();

                    view.update_in(
                        &mut cx,
                        |this: &mut RootView, _window, cx: &mut Context<RootView>| {
                            this.progress = AnalysisProgress::new(
                                AnalysisStage::RenderingGlyphs,
                                0,
                                total_faces,
                                format!("{total_faces} face を固定サイズへ描画しています"),
                            );
                            cx.notify();
                        },
                    )
                    .ok();

                    let (samples, excluded_faces): (Vec<FontSample>, Vec<ExcludedFace>) = cx
                        .background_executor()
                        .spawn({
                            let font_pipeline = font_pipeline.clone();
                            let request = request.clone();
                            let cache = cache.clone();
                            async move { font_pipeline.render_samples(&faces, &request, &cache) }
                        })
                        .await;

                    view.update_in(
                        &mut cx,
                        |this: &mut RootView, _window, cx: &mut Context<RootView>| {
                            this.progress = AnalysisProgress::new(
                                AnalysisStage::Embedding,
                                samples.len(),
                                total_faces,
                                format!(
                                    "Glyph encoder + morphology で {} 件を特徴化します",
                                    samples.len()
                                ),
                            );
                            cx.notify();
                        },
                    )
                    .ok();

                    let (device, device_label) = ml_pipeline.select_device();
                    let embedding_points: Vec<EmbeddingPoint> =
                        match cx
                            .background_executor()
                            .spawn({
                                let ml_pipeline = ml_pipeline.clone();
                                let request = request.clone();
                                let cache = cache.clone();
                                async move {
                                    ml_pipeline.embed_samples(&request, &samples, &cache, &device)
                                }
                            })
                            .await
                        {
                            Ok(points) => points,
                            Err(error) => {
                                view.update_in(
                                    &mut cx,
                                    |this: &mut RootView, _window, cx: &mut Context<RootView>| {
                                        this.fail(error.to_string(), cx);
                                    },
                                )
                                .ok();
                                return;
                            }
                        };

                    view.update_in(
                        &mut cx,
                        |this: &mut RootView, _window, cx: &mut Context<RootView>| {
                            this.progress = AnalysisProgress::new(
                                AnalysisStage::Reducing,
                                embedding_points.len(),
                                embedding_points.len(),
                                "Hybrid feature を UMAP で 2D 表示用に圧縮しています",
                            );
                            cx.notify();
                        },
                    )
                    .ok();

                    let projected: Vec<(f32, f32)> = match cx
                        .background_executor()
                        .spawn({
                            let ml_pipeline = ml_pipeline.clone();
                            let embedding_points = embedding_points.clone();
                            async move { ml_pipeline.project_embeddings(&embedding_points) }
                        })
                        .await
                    {
                        Ok(projected) => projected,
                        Err(error) => {
                            view.update_in(
                                &mut cx,
                                |this: &mut RootView, _window, cx: &mut Context<RootView>| {
                                    this.fail(error.to_string(), cx);
                                },
                            )
                            .ok();
                            return;
                        }
                    };

                    view.update_in(
                        &mut cx,
                        |this: &mut RootView, _window, cx: &mut Context<RootView>| {
                            this.progress = AnalysisProgress::new(
                                AnalysisStage::Clustering,
                                projected.len(),
                                projected.len(),
                                "HDBSCAN でクラスタリングしています",
                            );
                            cx.notify();
                        },
                    )
                    .ok();

                    let labels = cx
                        .background_executor()
                        .spawn({
                            let ml_pipeline = ml_pipeline.clone();
                            let embedding_points = embedding_points.clone();
                            async move { ml_pipeline.cluster_points(&embedding_points) }
                        })
                        .await;

                    let result = build_analysis_result(
                        request,
                        embedding_points,
                        projected,
                        labels,
                        excluded_faces,
                        total_faces,
                        device_label,
                    );

                    view.update_in(
                        &mut cx,
                        |this: &mut RootView, _window, cx: &mut Context<RootView>| {
                            this.finish(result, cx);
                        },
                    )
                    .ok();
                }
            },
        )
        .detach();
    }

    fn finish(&mut self, result: AnalysisResult, cx: &mut Context<Self>) {
        self.progress = AnalysisProgress::new(
            AnalysisStage::Finished,
            result.stats.rendered_faces,
            result.stats.scanned_faces,
            format!(
                "{} clusters / {} noise / {} excluded",
                result.stats.cluster_count, result.stats.noise_points, result.stats.excluded_faces
            ),
        );
        self.selected_face = result.points.first().map(|point| point.face.face_id);
        self.hovered_face = None;
        self.detail_face = self.selected_face;
        self.is_running = false;
        self.error_message = None;
        self.thumbnail_cache = build_thumbnail_cache(&result.points);
        self.result = Some(result);
        cx.notify();
    }

    fn fail(&mut self, message: String, cx: &mut Context<Self>) {
        self.progress = AnalysisProgress::new(AnalysisStage::Failed, 0, 0, message.clone());
        self.error_message = Some(message);
        self.is_running = false;
        self.detail_face = None;
        cx.notify();
    }

    fn on_plot_down(
        &mut self,
        event: &MouseDownEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.result.is_none() {
            return;
        }

        self.plot.drag_last = Some(event.position);
        self.plot.drag_moved = false;
        cx.notify();
    }

    fn on_plot_up(&mut self, event: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let dragged = self.plot.drag_moved;
        self.plot.drag_last = None;
        self.plot.drag_moved = false;

        if !dragged {
            let selection = self.hit_test_point(event.position);
            if self.selected_face != selection {
                self.selected_face = selection;
                self.detail_face = detail_face_after_selection(self.detail_face, selection);
                cx.notify();
            }
        }
    }

    fn on_plot_move(
        &mut self,
        event: &MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(last) = self.plot.drag_last {
            let delta = event.position - last;
            let dx = delta.x / px(1.0);
            let dy = delta.y / px(1.0);
            if dx.abs() > 0.5 || dy.abs() > 0.5 {
                self.plot.pan = self.plot.pan + delta;
                self.plot.drag_last = Some(event.position);
                self.plot.drag_moved = true;
                self.hovered_face = None;
                cx.notify();
            }
            return;
        }

        let hovered = self.hit_test_point(event.position);
        let next_detail_face = detail_face_after_hover(self.detail_face, hovered);
        if self.hovered_face != hovered || self.detail_face != next_detail_face {
            self.hovered_face = hovered;
            self.detail_face = next_detail_face;
            cx.notify();
        }
    }

    fn on_plot_scroll(
        &mut self,
        event: &ScrollWheelEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.result.is_none() {
            return;
        }

        let delta_y = match event.delta {
            ScrollDelta::Pixels(delta) => delta.y / px(1.0),
            ScrollDelta::Lines(delta) => delta.y * 20.0,
        };

        let Some(factor) = zoom_factor_for_delta(delta_y) else {
            return;
        };

        let previous_zoom = self.plot.zoom;
        let next_zoom = (previous_zoom * factor).clamp(0.35, 8.0);
        if let Some(bounds) = self.plot.bounds {
            self.plot.pan = anchored_pan_after_zoom(
                bounds,
                self.plot.pan,
                event.position,
                previous_zoom,
                next_zoom,
            );
        }
        self.plot.zoom = next_zoom;
        cx.notify();
    }

    fn plot_sources(&self) -> Vec<PlotSourcePoint> {
        self.result
            .as_ref()
            .map(|result| {
                result
                    .points
                    .iter()
                    .map(|point| PlotSourcePoint {
                        face_id: point.face.face_id,
                        x: point.x,
                        y: point.y,
                        cluster_id: point.cluster_id,
                        thumbnail: self
                            .thumbnail_for_face(point.face.face_id, point.preview.as_ref()),
                        selected: self.selected_face == Some(point.face.face_id),
                        hovered: self.hovered_face == Some(point.face.face_id),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    fn layout_plot_points(&self, bounds: Bounds<Pixels>) -> Vec<PlotPoint> {
        layout_plot_points(bounds, &self.plot, &self.plot_sources())
    }

    fn hit_test_point(&self, position: Point<Pixels>) -> Option<fontdb::ID> {
        let bounds = self.plot.bounds?;
        hit_test_plot_point(&self.layout_plot_points(bounds), position)
            .map(|plot_point| plot_point.face_id)
    }

    fn detail_point(&self) -> Option<&MapPoint> {
        let result = self.result.as_ref()?;
        let preferred = self
            .detail_face
            .or(self.hovered_face)
            .or(self.selected_face)?;
        result
            .points
            .iter()
            .find(|point| point.face.face_id == preferred)
            .or_else(|| result.points.first())
    }

    fn stat_value(&self, selector: impl FnOnce(&AnalysisStats) -> String) -> String {
        self.result
            .as_ref()
            .map(|result| selector(&result.stats))
            .unwrap_or_else(|| "0".to_string())
    }

    fn thumbnail_for_face(
        &self,
        face_id: fontdb::ID,
        fallback_image: &image::GrayImage,
    ) -> Arc<RenderImage> {
        self.thumbnail_cache
            .get(&face_id)
            .cloned()
            .unwrap_or_else(|| render_image_from_gray(fallback_image))
    }
}

impl Render for RootView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let status_text = self.progress.summary();
        let view = cx.entity();
        let plot_state = self.plot.clone();
        let source_points = self.plot_sources();
        let result_present = self.result.is_some();
        let detail_point = self.detail_point().cloned();
        let detail_preview = detail_point
            .as_ref()
            .map(|point| self.thumbnail_for_face(point.face.face_id, point.preview.as_ref()));
        let error_message = self.error_message.clone();
        let excluded_preview = self
            .result
            .as_ref()
            .map(|result| {
                result
                    .excluded_faces
                    .iter()
                    .take(5)
                    .map(|excluded| {
                        format!("{}: {}", excluded.face.display_name(), excluded.reason)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        div()
            .size_full()
            .bg(rgb(0xf1ede4))
            .text_color(rgb(0x1d211f))
            .child(
                div()
                    .size_full()
                    .flex()
                    .flex_col()
                    .p_6()
                    .gap_4()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .child(
                                div()
                                    .flex()
                                    .justify_between()
                                    .items_end()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_1()
                                            .child(
                                                div()
                                                    .text_size(px(30.0))
                                                    .font_weight(FontWeight::BOLD)
                                                    .child("Font Wandering"),
                                            )
                                            .child(
                                                div()
                                                    .text_color(rgb(0x5f655f))
                                                    .child("glyph encoder + morphology -> UMAP(view) -> HDBSCAN(high-dim)"),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .px_3()
                                            .py_2()
                                            .rounded_lg()
                                            .bg(rgb(0xe1dccd))
                                            .child(status_text),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .gap_3()
                                    .items_center()
                                    .child(div().flex_1().child(self.input.clone()))
                                    .child(
                                        div()
                                            .flex_none()
                                            .px_4()
                                            .py_2()
                                            .rounded_md()
                                            .cursor_pointer()
                                            .bg(if self.is_running {
                                                rgb(0x9ca29d)
                                            } else {
                                                rgb(0x1c6f52)
                                            })
                                            .text_color(white())
                                            .child(if self.is_running { "Running..." } else { "Analyze" })
                                            .on_mouse_down(
                                                MouseButton::Left,
                                                cx.listener(Self::analyze),
                                            ),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .gap_3()
                                    .children([
                                        stat_card(
                                            "Scanned",
                                            self.stat_value(|stats| stats.scanned_faces.to_string()),
                                        ),
                                        stat_card(
                                            "Rendered",
                                            self.stat_value(|stats| stats.rendered_faces.to_string()),
                                        ),
                                        stat_card(
                                            "Excluded",
                                            self.stat_value(|stats| stats.excluded_faces.to_string()),
                                        ),
                                        stat_card(
                                            "Clusters",
                                            self.stat_value(|stats| stats.cluster_count.to_string()),
                                        ),
                                        stat_card(
                                            "Device",
                                            self.stat_value(|stats| stats.device_label.clone()),
                                        ),
                                    ]),
                            ),
                    )
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .gap_4()
                            .child(
                                div()
                                    .flex_1()
                                    .rounded_xl()
                                    .border_1()
                                    .border_color(rgb(0xd2cab7))
                                    .bg(rgb(0xfcfbf7))
                                    .overflow_hidden()
                                    .child(
                                        div()
                                            .size_full()
                                            .relative()
                                            .cursor_pointer()
                                            .on_mouse_down(
                                                MouseButton::Left,
                                                cx.listener(Self::on_plot_down),
                                            )
                                            .on_mouse_up(
                                                MouseButton::Left,
                                                cx.listener(Self::on_plot_up),
                                            )
                                            .on_mouse_up_out(
                                                MouseButton::Left,
                                                cx.listener(Self::on_plot_up),
                                            )
                                            .on_mouse_move(cx.listener(Self::on_plot_move))
                                            .on_scroll_wheel(cx.listener(Self::on_plot_scroll))
                                            .child(
                                                canvas(
                                                    move |bounds, _window, _cx| {
                                                        layout_plot_points(bounds, &plot_state, &source_points)
                                                    },
                                                    move |bounds, plotted, window, cx| {
                                                        let _ = view.update(cx, |this, _cx| {
                                                            this.plot.bounds = Some(bounds);
                                                        });

                                                        window.paint_quad(fill(bounds, rgb(0xfaf7ee)));
                                                        window.paint_quad(outline(
                                                            bounds,
                                                            rgb(0xd4c8ae),
                                                            Default::default(),
                                                        ));

                                                        for plot_point in plotted {
                                                            let corner_radius = plot_corner_radius(plot_point.bounds);
                                                            let _ = window.paint_image(
                                                                plot_point.bounds,
                                                                Corners::all(corner_radius),
                                                                plot_point.thumbnail.clone(),
                                                                0,
                                                                false,
                                                            );

                                                            window.paint_quad(quad(
                                                                plot_point.bounds,
                                                                corner_radius,
                                                                transparent_black(),
                                                                px(THUMBNAIL_BORDER_WIDTH),
                                                                cluster_color(plot_point.cluster_id),
                                                                Default::default(),
                                                            ));

                                                            if plot_point.selected || plot_point.hovered {
                                                                let ring_bounds =
                                                                    expand_bounds(plot_point.bounds, px(THUMBNAIL_RING_GAP));
                                                                window.paint_quad(quad(
                                                                    ring_bounds,
                                                                    plot_corner_radius(ring_bounds),
                                                                    transparent_black(),
                                                                    px(1.8),
                                                                    rgb(0x20362b),
                                                                    Default::default(),
                                                                ));
                                                            }
                                                        }
                                                    },
                                                )
                                                .size_full(),
                                            )
                                            .when(!result_present, |this| {
                                                this.child(
                                                    div()
                                                        .absolute()
                                                        .inset_0()
                                                        .flex()
                                                        .items_center()
                                                        .justify_center()
                                                        .text_color(rgb(0x777f79))
                                                        .child("Analyze を押すと 2D map を描画します"),
                                                )
                                            }),
                                    ),
                            )
                            .child(
                                div()
                                    .id("details-panel")
                                    .w(px(340.0))
                                    .rounded_xl()
                                    .border_1()
                                    .border_color(rgb(0xd2cab7))
                                    .bg(rgb(0xf8f4eb))
                                    .p_4()
                                    .flex()
                                    .flex_col()
                                    .overflow_y_scroll()
                                    .scrollbar_width(px(12.0))
                                    .gap_3()
                                    .child(
                                        div()
                                            .text_size(px(20.0))
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .child("Details"),
                                    )
                                    .child(match detail_point {
                                        Some(point) => {
                                            let preview = detail_preview
                                                .clone()
                                                .unwrap_or_else(|| render_image_from_gray(point.preview.as_ref()));
                                            div()
                                                .flex()
                                                .flex_col()
                                                .gap_3()
                                                .child(
                                                    img(preview)
                                                        .w_full()
                                                        .h(px(220.0))
                                                        .rounded_lg()
                                                        .border_1()
                                                        .border_color(rgb(0xd2cab7)),
                                                )
                                                .child(metadata_row("Family", &point.face.family))
                                                .child(metadata_row("Style", &point.face.style))
                                                .child(metadata_row("StyleKind", &point.face.style_kind.to_string()))
                                                .child(metadata_row(
                                                    "Weight",
                                                    &format!("{} ({})", point.face.weight_label(), point.face.weight_value),
                                                ))
                                                .child(metadata_row(
                                                    "Stretch",
                                                    &format!(
                                                        "{} ({})",
                                                        point.face.stretch_label(),
                                                        point.face.stretch_value
                                                    ),
                                                ))
                                                .child(metadata_row("PostScript", &point.face.post_script_name))
                                                .child(metadata_row("Cluster", &point.cluster_id.to_string()))
                                                .child(metadata_row(
                                                    "UMAP",
                                                    &format!("{:.3}, {:.3}", point.x, point.y),
                                                ))
                                                .child(metadata_row(
                                                    "Embedding",
                                                    &format!("{} dims", point.embedding.len()),
                                                ))
                                                .child(metadata_row("Source", &point.face.source_label()))
                                                .into_any_element()
                                        }
                                        None => div()
                                            .rounded_lg()
                                            .bg(rgb(0xeee7d6))
                                            .p_3()
                                            .text_color(rgb(0x6a7068))
                                            .child("点を hover / click すると詳細を表示します")
                                            .into_any_element(),
                                    })
                                    .when(!excluded_preview.is_empty(), |this| {
                                        this.child(
                                            div()
                                                .mt_2()
                                                .flex()
                                                .flex_col()
                                                .gap_2()
                                                .child(
                                                    div()
                                                        .text_size(px(16.0))
                                                        .font_weight(FontWeight::SEMIBOLD)
                                                        .child("Excluded Samples"),
                                                )
                                                .children(
                                                    excluded_preview
                                                        .into_iter()
                                                        .map(|line| {
                                                            div()
                                                                .text_size(px(13.0))
                                                                .text_color(rgb(0x6a7068))
                                                                .child(line)
                                                                .into_any_element()
                                                        })
                                                        .collect::<Vec<_>>(),
                                                ),
                                        )
                                    })
                                    .when_some(error_message, |this, message| {
                                        this.child(
                                            div()
                                                .rounded_lg()
                                                .bg(rgb(0xf0d7cd))
                                                .text_color(rgb(0x7a2417))
                                                .p_3()
                                                .child(message),
                                        )
                                    }),
                            ),
                    ),
            )
    }
}

fn build_analysis_result(
    request: AnalysisRequest,
    embedding_points: Vec<EmbeddingPoint>,
    projected: Vec<(f32, f32)>,
    labels: Vec<i32>,
    excluded_faces: Vec<ExcludedFace>,
    scanned_faces: usize,
    device_label: String,
) -> AnalysisResult {
    let embedding_dim = embedding_points
        .first()
        .map(|point| point.embedding.len())
        .unwrap_or(0);

    let points = embedding_points
        .into_iter()
        .zip(projected)
        .zip(labels)
        .map(|((point, (x, y)), cluster_id)| MapPoint {
            face: point.face,
            x,
            y,
            cluster_id,
            preview: point.image,
            embedding: point.embedding,
        })
        .collect::<Vec<_>>();

    let cluster_count = points
        .iter()
        .filter_map(|point| (point.cluster_id >= 0).then_some(point.cluster_id))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let noise_points = points.iter().filter(|point| point.cluster_id < 0).count();

    let rendered_faces = points.len();
    let excluded_count = excluded_faces.len();

    AnalysisResult {
        request,
        points,
        excluded_faces,
        stats: AnalysisStats {
            scanned_faces,
            rendered_faces,
            excluded_faces: excluded_count,
            cluster_count,
            noise_points,
            embedding_dim,
            device_label,
        },
    }
}

fn layout_plot_points(
    bounds: Bounds<Pixels>,
    plot_state: &PlotState,
    source_points: &[PlotSourcePoint],
) -> Vec<PlotPoint> {
    if source_points.is_empty() {
        return Vec::new();
    }

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for point in source_points {
        min_x = min_x.min(point.x);
        max_x = max_x.max(point.x);
        min_y = min_y.min(point.y);
        max_y = max_y.max(point.y);
    }

    let span_x = (max_x - min_x).max(1e-3);
    let span_y = (max_y - min_y).max(1e-3);
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    let scale = plot_data_scale(bounds, plot_state.zoom, span_x, span_y);
    let center = bounds.center() + plot_state.pan;
    let base_side = plot_thumbnail_side(bounds);

    let mut plotted = source_points
        .iter()
        .map(|source| {
            let position = point(
                center.x + px((source.x - center_x) * scale),
                center.y - px((source.y - center_y) * scale),
            );
            let side = base_side * thumbnail_scale(source);
            let bounds = Bounds::new(
                point(position.x - side * 0.5, position.y - side * 0.5),
                size(side, side),
            );

            PlotPoint {
                face_id: source.face_id,
                bounds,
                cluster_id: source.cluster_id,
                thumbnail: source.thumbnail.clone(),
                selected: source.selected,
                hovered: source.hovered,
                z_order: z_order_for_source(source),
            }
        })
        .collect::<Vec<_>>();

    plotted.sort_by_key(|point| point.z_order);
    plotted
}

fn plot_data_scale(bounds: Bounds<Pixels>, zoom: f32, span_x: f32, span_y: f32) -> f32 {
    let usable_width = ((bounds.size.width - px(PLOT_PADDING * 2.0)) / px(1.0)).max(1.0);
    let usable_height = ((bounds.size.height - px(PLOT_PADDING * 2.0)) / px(1.0)).max(1.0);
    (usable_width / span_x).min(usable_height / span_y) * zoom
}

fn zoom_factor_for_delta(delta_y: f32) -> Option<f32> {
    if delta_y > 0.0 {
        Some(0.92)
    } else if delta_y < 0.0 {
        Some(1.08)
    } else {
        None
    }
}

fn anchored_pan_after_zoom(
    bounds: Bounds<Pixels>,
    current_pan: Point<Pixels>,
    anchor: Point<Pixels>,
    previous_zoom: f32,
    next_zoom: f32,
) -> Point<Pixels> {
    if previous_zoom <= f32::EPSILON {
        return current_pan;
    }

    let zoom_ratio = next_zoom / previous_zoom;
    let anchor_offset = anchor - (bounds.center() + current_pan);
    current_pan + anchor_offset * (1.0 - zoom_ratio)
}

fn detail_face_after_hover<T: Copy>(
    current_detail: Option<T>,
    hovered_face: Option<T>,
) -> Option<T> {
    hovered_face.or(current_detail)
}

fn detail_face_after_selection<T: Copy>(
    current_detail: Option<T>,
    selected_face: Option<T>,
) -> Option<T> {
    selected_face.or(current_detail)
}

fn plot_thumbnail_side(bounds: Bounds<Pixels>) -> Pixels {
    let min_axis = ((bounds.size.width / px(1.0)).min(bounds.size.height / px(1.0))).max(1.0);
    px((min_axis * THUMBNAIL_BASE_RATIO).clamp(THUMBNAIL_MIN_SIDE, THUMBNAIL_MAX_SIDE))
}

fn thumbnail_scale(source: &PlotSourcePoint) -> f32 {
    if source.selected {
        THUMBNAIL_SELECTED_SCALE
    } else if source.hovered {
        THUMBNAIL_HOVER_SCALE
    } else {
        1.0
    }
}

fn z_order_for_source(source: &PlotSourcePoint) -> u8 {
    if source.selected {
        2
    } else if source.hovered {
        1
    } else {
        0
    }
}

fn plot_corner_radius(bounds: Bounds<Pixels>) -> Pixels {
    let side = (bounds.size.width / px(1.0)).min(bounds.size.height / px(1.0));
    px((side * 0.18).clamp(THUMBNAIL_CORNER_RADIUS, THUMBNAIL_CORNER_RADIUS + 2.0))
}

fn expand_bounds(bounds: Bounds<Pixels>, expansion: Pixels) -> Bounds<Pixels> {
    Bounds::new(
        point(bounds.origin.x - expansion, bounds.origin.y - expansion),
        size(
            bounds.size.width + expansion * 2.0,
            bounds.size.height + expansion * 2.0,
        ),
    )
}

fn hit_test_plot_point(plotted: &[PlotPoint], position: Point<Pixels>) -> Option<&PlotPoint> {
    plotted
        .iter()
        .rev()
        .find(|plot_point| plot_point.bounds.contains(&position))
}

fn cluster_color(cluster_id: i32) -> gpui::Hsla {
    if cluster_id < 0 {
        return rgb(0x6f6f6f).into();
    }

    const PALETTE: [u32; 8] = [
        0x2c6e49, 0xc44536, 0x26547c, 0xd68c45, 0x5b5f97, 0x8e5572, 0x1f7a8c, 0xb56576,
    ];
    rgb(PALETTE[cluster_id as usize % PALETTE.len()]).into()
}

fn stat_card(label: &str, value: String) -> gpui::AnyElement {
    div()
        .min_w(px(110.0))
        .px_3()
        .py_2()
        .rounded_lg()
        .bg(rgb(0xe8e1d0))
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_size(px(11.0))
                .text_color(rgb(0x6f756d))
                .child(label.to_string()),
        )
        .child(
            div()
                .text_size(px(18.0))
                .font_weight(FontWeight::SEMIBOLD)
                .child(value),
        )
        .into_any_element()
}

fn metadata_row(label: &str, value: &str) -> gpui::AnyElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_size(px(11.0))
                .text_color(rgb(0x6f756d))
                .child(label.to_string()),
        )
        .child(
            div()
                .text_size(px(14.0))
                .line_height(px(20.0))
                .child(value.to_string()),
        )
        .into_any_element()
}

fn build_thumbnail_cache(points: &[MapPoint]) -> HashMap<fontdb::ID, Arc<RenderImage>> {
    points
        .iter()
        .map(|point| {
            (
                point.face.face_id,
                render_image_from_gray(point.preview.as_ref()),
            )
        })
        .collect()
}

fn render_image_from_gray(image: &image::GrayImage) -> Arc<RenderImage> {
    let mut buffer =
        ImageBuffer::<Rgba<u8>, Vec<u8>>::from_fn(image.width(), image.height(), |x, y| {
            let value = image.get_pixel(x, y).0[0];
            Rgba([value, value, value, 255])
        });

    for pixel in buffer.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }

    Arc::new(RenderImage::new(smallvec![Frame::new(buffer)]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_thumbnail(level: u8) -> Arc<RenderImage> {
        let image = image::GrayImage::from_fn(8, 8, |_, _| image::Luma([level]));
        render_image_from_gray(&image)
    }

    fn source_point(selected: bool, hovered: bool) -> PlotSourcePoint {
        PlotSourcePoint {
            face_id: fontdb::ID::dummy(),
            x: 0.0,
            y: 0.0,
            cluster_id: 1,
            thumbnail: sample_thumbnail(192),
            selected,
            hovered,
        }
    }

    #[test]
    fn render_image_from_gray_creates_single_frame_image() {
        let image = image::GrayImage::from_fn(4, 3, |x, y| image::Luma([(x as u8) ^ (y as u8)]));
        let rendered = render_image_from_gray(&image);

        assert_eq!(rendered.frame_count(), 1);
        assert_eq!(rendered.size(0).width.0, 4);
        assert_eq!(rendered.size(0).height.0, 3);
        assert_eq!(rendered.as_bytes(0).unwrap().len(), 4 * 3 * 4);
    }

    #[test]
    fn layout_plot_points_includes_bounds_and_thumbnail() {
        let points = layout_plot_points(
            Bounds::new(point(px(0.0), px(0.0)), size(px(600.0), px(420.0))),
            &PlotState::default(),
            &[PlotSourcePoint {
                face_id: fontdb::ID::dummy(),
                x: 1.0,
                y: -1.0,
                cluster_id: 3,
                thumbnail: sample_thumbnail(144),
                selected: false,
                hovered: false,
            }],
        );

        assert_eq!(points.len(), 1);
        assert!(points[0].bounds.size.width > px(0.0));
        assert!(points[0].bounds.size.height > px(0.0));
        assert_eq!(points[0].thumbnail.frame_count(), 1);
        assert!(points[0].bounds.contains(&points[0].bounds.center()));
    }

    #[test]
    fn hit_test_plot_point_prefers_frontmost_thumbnail() {
        let shared_bounds = Bounds::new(point(px(10.0), px(10.0)), size(px(32.0), px(32.0)));
        let plotted = vec![
            PlotPoint {
                face_id: fontdb::ID::dummy(),
                bounds: shared_bounds,
                cluster_id: 0,
                thumbnail: sample_thumbnail(64),
                selected: false,
                hovered: false,
                z_order: 0,
            },
            PlotPoint {
                face_id: fontdb::ID::dummy(),
                bounds: shared_bounds,
                cluster_id: 1,
                thumbnail: sample_thumbnail(200),
                selected: true,
                hovered: false,
                z_order: 2,
            },
        ];

        let hit = hit_test_plot_point(&plotted, point(px(20.0), px(20.0))).unwrap();
        assert!(hit.selected);
    }

    #[test]
    fn layout_plot_points_sorts_hovered_and_selected_last() {
        let plotted = layout_plot_points(
            Bounds::new(point(px(0.0), px(0.0)), size(px(640.0), px(480.0))),
            &PlotState::default(),
            &[
                source_point(true, false),
                source_point(false, false),
                source_point(false, true),
            ],
        );

        assert_eq!(plotted.len(), 3);
        assert!(!plotted[0].selected && !plotted[0].hovered);
        assert!(plotted[1].hovered);
        assert!(plotted[2].selected);
        assert!(plotted[2].z_order >= plotted[1].z_order);
    }

    #[test]
    fn anchored_pan_after_zoom_keeps_cursor_target_stationary() {
        let bounds = Bounds::new(point(px(0.0), px(0.0)), size(px(600.0), px(420.0)));
        let source_points = vec![
            PlotSourcePoint {
                face_id: fontdb::ID::dummy(),
                x: -1.0,
                y: -0.6,
                cluster_id: 0,
                thumbnail: sample_thumbnail(80),
                selected: false,
                hovered: false,
            },
            PlotSourcePoint {
                face_id: fontdb::ID::dummy(),
                x: 1.4,
                y: 0.8,
                cluster_id: 1,
                thumbnail: sample_thumbnail(180),
                selected: false,
                hovered: false,
            },
        ];

        let mut plot_state = PlotState::default();
        let before = layout_plot_points(bounds, &plot_state, &source_points);
        let anchor = before[1].bounds.center();

        let next_zoom = 1.8;
        plot_state.pan =
            anchored_pan_after_zoom(bounds, plot_state.pan, anchor, plot_state.zoom, next_zoom);
        plot_state.zoom = next_zoom;

        let after = layout_plot_points(bounds, &plot_state, &source_points);
        let anchored_center = after[1].bounds.center();

        assert!((anchored_center.x - anchor.x).abs() <= px(0.01));
        assert!((anchored_center.y - anchor.y).abs() <= px(0.01));
    }

    #[test]
    fn zoom_factor_for_delta_ignores_zero_delta() {
        assert_eq!(zoom_factor_for_delta(10.0), Some(0.92));
        assert_eq!(zoom_factor_for_delta(-10.0), Some(1.08));
        assert_eq!(zoom_factor_for_delta(0.0), None);
    }

    #[test]
    fn detail_face_after_hover_keeps_previous_face_when_hover_temporarily_clears() {
        assert_eq!(detail_face_after_hover(Some(7_u8), Some(9_u8)), Some(9_u8));
        assert_eq!(detail_face_after_hover(Some(7_u8), None), Some(7_u8));
        assert_eq!(detail_face_after_hover(None::<u8>, None), None);
    }

    #[test]
    fn detail_face_after_selection_uses_new_selection_but_keeps_existing_on_blank_click() {
        assert_eq!(
            detail_face_after_selection(Some(3_u8), Some(5_u8)),
            Some(5_u8)
        );
        assert_eq!(detail_face_after_selection(Some(3_u8), None), Some(3_u8));
        assert_eq!(detail_face_after_selection(None::<u8>, None), None);
    }
}
