use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use fontdb::ID;

use crate::models::{RenderSpec, SharedGrayImage};

#[derive(Clone, Debug, Default)]
pub struct AnalysisCache {
    rendered: Arc<Mutex<HashMap<RenderCacheKey, SharedGrayImage>>>,
    embeddings: Arc<Mutex<HashMap<EmbeddingCacheKey, Arc<Vec<f32>>>>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct RenderCacheKey {
    text: String,
    face_id: ID,
    render_spec: RenderSpec,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct EmbeddingCacheKey {
    text: String,
    face_id: ID,
    render_spec: RenderSpec,
    model_revision: String,
}

impl AnalysisCache {
    pub fn rendered(
        &self,
        text: &str,
        face_id: ID,
        render_spec: &RenderSpec,
    ) -> Option<SharedGrayImage> {
        self.rendered
            .lock()
            .expect("render cache poisoned")
            .get(&RenderCacheKey {
                text: text.to_string(),
                face_id,
                render_spec: render_spec.clone(),
            })
            .cloned()
    }

    pub fn put_rendered(
        &self,
        text: &str,
        face_id: ID,
        render_spec: &RenderSpec,
        image: SharedGrayImage,
    ) {
        self.rendered.lock().expect("render cache poisoned").insert(
            RenderCacheKey {
                text: text.to_string(),
                face_id,
                render_spec: render_spec.clone(),
            },
            image,
        );
    }

    pub fn embedding(
        &self,
        text: &str,
        face_id: ID,
        render_spec: &RenderSpec,
        model_revision: &str,
    ) -> Option<Arc<Vec<f32>>> {
        self.embeddings
            .lock()
            .expect("embedding cache poisoned")
            .get(&EmbeddingCacheKey {
                text: text.to_string(),
                face_id,
                render_spec: render_spec.clone(),
                model_revision: model_revision.to_string(),
            })
            .cloned()
    }

    pub fn put_embedding(
        &self,
        text: &str,
        face_id: ID,
        render_spec: &RenderSpec,
        model_revision: &str,
        embedding: Arc<Vec<f32>>,
    ) {
        self.embeddings
            .lock()
            .expect("embedding cache poisoned")
            .insert(
                EmbeddingCacheKey {
                    text: text.to_string(),
                    face_id,
                    render_spec: render_spec.clone(),
                    model_revision: model_revision.to_string(),
                },
                embedding,
            );
    }
}
