use gpui::{App, AppContext, Application, Bounds, WindowBounds, WindowOptions, px, size};

use crate::ui::{RootView, bind_keys};

pub fn run() {
    Application::new().run(|cx: &mut App| {
        bind_keys(cx);
        let bounds = Bounds::centered(None, size(px(1440.0), px(960.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(gpui::TitlebarOptions {
                    title: Some("Font Wandering".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            |window, cx| cx.new(|cx| RootView::new(window, cx)),
        )
        .expect("failed to open main window");

        cx.activate(true);
    });
}
