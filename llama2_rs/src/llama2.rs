extern crate rand;
use rand::thread_rng;
use std::{
    collections::HashMap,
    fs::File,
    io::{stdout, BufReader, Read, Result, Write},
    mem,
    ops::Range,
    time::Instant,
};

use crate::ops::*;

#[derive(Debug, Default)]
pub struct Config {
    dim: i32,         // transformer dimension
    hidden_dim: i32,  // for ffn layers
    n_layers: i32,    // number of layers
    n_heads: i32,     // number of query heads
    _n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    _vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32,     // max sequence length
}

const CONFIG_SIZE: usize = mem::size_of::<Config>();

impl Config {
    pub fn new_from_file(path: &str) -> Result<Self> {
        let mut file = BufReader::new(File::open(path).expect("Cannot open config file"));
        let mut buffer = [0u8; CONFIG_SIZE];
        file.read_exact(&mut buffer).unwrap();
        let config = unsafe { mem::transmute::<[u8; CONFIG_SIZE], Config>(buffer) };
        Ok(config)
    }
    pub fn dim(&self) -> usize {
        self.dim as usize
    }
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim as usize
    }
    pub fn n_layers(&self) -> usize {
        self.n_layers as usize
    }
    pub fn n_heads(&self) -> usize {
        self.n_heads as usize
    }
    pub fn seq_len(&self) -> usize {
        self.seq_len as usize
    }
    pub fn vocab_size(&self) -> usize {
        if self._vocab_size < 0 {
            (-self._vocab_size) as usize
        } else {
            self._vocab_size as usize
        }
    }
    pub fn head_size(&self) -> usize {
        (self.dim / self.n_heads) as usize
    }
    pub fn is_shared(&self) -> bool {
        self._vocab_size > 0
    }
}

fn read_vec<TValue: Clone>(file: &mut BufReader<File>, size: usize) -> Vec<TValue> {
    let mut buffer = vec![0u8; size * mem::size_of::<TValue>()];
    file.read_exact(&mut buffer).unwrap();
    let vec: Vec<TValue> =
        unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const TValue, size).to_vec() };
    vec
}

#[inline(always)]
fn slice(slice: &[T], start: usize, delta: usize) -> &[T] {
    &slice[start..start + delta]
}

#[inline(always)]
fn slice_mut(slice: &mut [T], start: usize, delta: usize) -> &mut [T] {
    &mut slice[start..start + delta]
}

#[derive(Debug, Default)]
pub struct TransformerWeights {
    // token embedding table
    _token_embedding_table: Vec<T>, // (vocab_size, dim)
    // weights for rmsnorms attention
    _rms_att_weight: Vec<T>, // (layer, dim) rmsnorm weights
    // weights for matmuls
    _wq: Vec<T>, // (layer, dim, dim)
    _wk: Vec<T>, // (layer, dim, dim)
    _wv: Vec<T>, // (layer, dim, dim)
    _wo: Vec<T>, // (layer, dim, dim)
    // weights for rmsnorms ffn
    _rms_ffn_weight: Vec<T>, // (layer, dim)
    // weights for ffn
    _w1: Vec<T>, // (layer, hidden_dim, dim)
    _w2: Vec<T>, // (layer, dim, hidden_dim)
    _w3: Vec<T>, // (layer, hidden_dim, dim)
    // final rmsnorm
    _rms_final_weight: Vec<T>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    _freq_cis_real: Vec<T>, // (seq_len, dim/2)
    _freq_cis_imag: Vec<T>, // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    _wcls: Option<Vec<T>>,
    _is_shared_wcls: bool, // Whether wcls is shared with token embedding table.

    // Ranges for lazy loading.
    _is_lazy: bool,
    _token_embedding_table_range: Range<usize>,
    _rms_att_weight_range: Range<usize>,
    _wq_range: Range<usize>,
    _wv_range: Range<usize>,
    _wk_range: Range<usize>,
    _wo_range: Range<usize>,
    _rms_ffn_weight_range: Range<usize>,
    _w1_range: Range<usize>,
    _w2_range: Range<usize>,
    _w3_range: Range<usize>,
    _rms_final_weight_range: Range<usize>,
    _freq_cis_real_range: Range<usize>,
    _freq_cis_imag_range: Range<usize>,
    _wcls_range: Range<usize>,
    _owner: MmapOwner,
}

struct MmapOwner {
    mmap: Option<memmap2::Mmap>,
}
impl MmapOwner {
    pub fn is_none(&self) -> bool {
        self.mmap.is_none()
    }
    // Casts block to T.
    pub fn cast_block(&self, range: &Range<usize>) -> &[T] {
        let bytes = &self.mmap.as_ref().unwrap()[range.clone()];
        unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const T,
                bytes.len() / mem::size_of::<T>(),
            )
        }
    }
}
impl Default for MmapOwner {
    fn default() -> Self {
        Self { mmap: None }
    }
}
impl std::fmt::Debug for MmapOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = if self.is_none() { "none" } else { "some" };
        write!(f, "mmap: {}", s)
    }
}

impl TransformerWeights {
    pub fn new(path: &str, config: &Config, is_mmap: bool) -> Result<Self> {
        if is_mmap {
            TransformerWeights::_new_mmap_from_file(path, &config)
        } else {
            TransformerWeights::_new_from_file(path, &config)
        }
    }

    fn _new_from_file(path: &str, config: &Config) -> Result<Self> {
        let mut file = BufReader::new(File::open(path).expect("Cannot open weights file"));
        file.seek_relative(CONFIG_SIZE as i64)?; // Skip config size at the head.

        let weights = Self {
            // Initialize order depends on storage.
            _token_embedding_table: read_vec(&mut file, config.vocab_size() * config.dim()),
            _rms_att_weight: read_vec(&mut file, config.n_layers() * config.dim()),
            _wq: read_vec(&mut file, config.n_layers() * config.dim() * config.dim()),
            _wk: read_vec(&mut file, config.n_layers() * config.dim() * config.dim()),
            _wv: read_vec(&mut file, config.n_layers() * config.dim() * config.dim()),
            _wo: read_vec(&mut file, config.n_layers() * config.dim() * config.dim()),
            _rms_ffn_weight: read_vec(&mut file, config.n_layers() * config.dim()),
            _w1: read_vec(
                &mut file,
                config.n_layers() * config.hidden_dim() * config.dim(),
            ),
            _w2: read_vec(
                &mut file,
                config.n_layers() * config.dim() * config.hidden_dim(),
            ),
            _w3: read_vec(
                &mut file,
                config.n_layers() * config.hidden_dim() * config.dim(),
            ),
            _rms_final_weight: read_vec(&mut file, config.dim()),
            _freq_cis_real: read_vec(&mut file, config.seq_len() * config.head_size() / 2),
            _freq_cis_imag: read_vec(&mut file, config.seq_len() * config.head_size() / 2),
            // If shared, wcls shares weights with token_embedding_table.
            _wcls: (!config.is_shared())
                .then(|| read_vec(&mut file, config.vocab_size() * config.dim())),
            _is_shared_wcls: config.is_shared(),
            _is_lazy: false,
            ..Default::default()
        };
        Ok(weights)
    }

    fn _new_mmap_from_file(path: &str, config: &Config) -> Result<Self> {
        let file = File::open(path).expect("Cannot open weights file");
        let owner = MmapOwner {
            mmap: Some(unsafe { memmap2::MmapOptions::new().map(&file) }?),
        };
        let mut offset: usize = CONFIG_SIZE;
        let type_size = mem::size_of::<T>();
        let lazy_range = |cur_offset: &mut usize, size: usize| -> Range<usize> {
            let r = Range {
                start: *cur_offset,
                end: (*cur_offset) + size * type_size,
            };
            *cur_offset = r.end;
            r
        };
        let mut weights = Self {
            _token_embedding_table_range: lazy_range(
                &mut offset,
                config.vocab_size() * config.dim(),
            ),
            _rms_att_weight_range: lazy_range(&mut offset, config.n_layers() * config.dim()),
            _wq_range: lazy_range(&mut offset, config.n_layers() * config.dim() * config.dim()),
            _wk_range: lazy_range(&mut offset, config.n_layers() * config.dim() * config.dim()),
            _wv_range: lazy_range(&mut offset, config.n_layers() * config.dim() * config.dim()),
            _wo_range: lazy_range(&mut offset, config.n_layers() * config.dim() * config.dim()),
            _rms_ffn_weight_range: lazy_range(&mut offset, config.n_layers() * config.dim()),
            _w1_range: lazy_range(
                &mut offset,
                config.n_layers() * config.hidden_dim() * config.dim(),
            ),
            _w2_range: lazy_range(
                &mut offset,
                config.n_layers() * config.dim() * config.hidden_dim(),
            ),
            _w3_range: lazy_range(
                &mut offset,
                config.n_layers() * config.hidden_dim() * config.dim(),
            ),
            _rms_final_weight_range: lazy_range(&mut offset, config.dim()),
            _freq_cis_real_range: lazy_range(
                &mut offset,
                config.seq_len() * config.head_size() / 2,
            ),
            _freq_cis_imag_range: lazy_range(
                &mut offset,
                config.seq_len() * config.head_size() / 2,
            ),
            _is_shared_wcls: config.is_shared(),
            _is_lazy: true,
            _owner: owner,
            ..Default::default()
        };
        weights._wcls_range = if weights._is_shared_wcls {
            weights._token_embedding_table_range.clone()
        } else {
            lazy_range(&mut offset, config.vocab_size() * config.dim())
        };
        Ok(weights)
    }

    #[inline(always)]
    fn get_maybe_lazy<'ret, 's: 'ret, 'data: 'ret>(
        &'s self,
        data: &'data [T],
        range: &Range<usize>,
    ) -> &'ret [T] {
        if self._is_lazy {
            &self._owner.cast_block(range)
        } else {
            data
        }
    }

    pub fn token_embedding_table(&self) -> &[T] {
        self.get_maybe_lazy(
            &self._token_embedding_table,
            &self._token_embedding_table_range,
        )
    }
    pub fn rms_att_weight(&self) -> &[T] {
        self.get_maybe_lazy(&self._rms_att_weight, &self._rms_att_weight_range)
    }
    pub fn wq(&self) -> &[T] {
        self.get_maybe_lazy(&self._wq, &self._wq_range)
    }
    pub fn wk(&self) -> &[T] {
        self.get_maybe_lazy(&self._wk, &self._wk_range)
    }
    pub fn wv(&self) -> &[T] {
        self.get_maybe_lazy(&self._wv, &self._wv_range)
    }
    pub fn wo(&self) -> &[T] {
        self.get_maybe_lazy(&self._wo, &self._wo_range)
    }
    pub fn rms_ffn_weight(&self) -> &[T] {
        self.get_maybe_lazy(&self._rms_ffn_weight, &self._rms_ffn_weight_range)
    }
    pub fn w1(&self) -> &[T] {
        self.get_maybe_lazy(&self._w1, &self._w1_range)
    }
    pub fn w2(&self) -> &[T] {
        self.get_maybe_lazy(&self._w2, &self._w2_range)
    }
    pub fn w3(&self) -> &[T] {
        self.get_maybe_lazy(&self._w3, &self._w3_range)
    }
    pub fn rms_final_weight(&self) -> &[T] {
        self.get_maybe_lazy(&self._rms_final_weight, &self._rms_final_weight_range)
    }
    pub fn freq_cis_real(&self) -> &[T] {
        self.get_maybe_lazy(&self._freq_cis_real, &self._freq_cis_real_range)
    }
    pub fn freq_cis_imag(&self) -> &[T] {
        self.get_maybe_lazy(&self._freq_cis_imag, &self._freq_cis_imag_range)
    }
    pub fn wcls(&self) -> &[T] {
        if self._is_lazy {
            self._owner.cast_block(&self._wcls_range)
        } else {
            if self._is_shared_wcls {
                &self._token_embedding_table
            } else {
                self._wcls.as_ref().unwrap()
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    vocab: Vec<String>,
    vocab_map: HashMap<String, usize>, // TODO: Optimize storage size.
    scores: Vec<f32>,
    _max_token_length: i32,
}
pub const BOS: usize = 1;

impl Vocabulary {
    pub fn new_from_file(path: &str, config: &Config) -> Result<Self> {
        let mut file = BufReader::new(File::open(path).expect("Cannot open tokenizer file"));
        let mut i32buf = [0u8; mem::size_of::<i32>()];
        let mut f32buf = [0u8; mem::size_of::<f32>()];
        file.read_exact(&mut i32buf)?;
        let max_token_length = i32::from_le_bytes(i32buf);

        let mut vocab: Vec<String> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for _ in 0..config.vocab_size() {
            file.read_exact(&mut f32buf)?;
            scores.push(f32::from_le_bytes(f32buf));

            file.read_exact(&mut i32buf)?;
            let len = i32::from_ne_bytes(i32buf);
            let mut strbuff = vec![0u8; len as usize];
            file.read_exact(&mut strbuff)?;
            let str = String::from_utf8(strbuff).expect("string cannot be utf8");
            vocab.push(str);
        }
        let vocab_map = HashMap::<String, usize>::from_iter(
            vocab
                .iter()
                .enumerate()
                .map(|(i, str)| (String::from(str), i)),
        );
        Ok(Self {
            vocab,
            vocab_map,
            scores,
            _max_token_length: max_token_length,
        })
    }

    pub fn get_token_string(&self, token: usize, is_start: bool) -> &str {
        let next_str = &self.vocab[token];
        let next_str = if is_start && next_str.starts_with(' ') {
            next_str.trim_start()
        } else {
            next_str
        };
        next_str
    }

    pub fn bpe_encode(&self, text: &str) -> Vec<usize> {
        let mut tokens: Vec<usize> = vec![];
        if text.is_empty() {
            return tokens;
        }
        for c in text.chars() {
            let char_str = String::from(c);
            let t = self
                .vocab_map
                .get(&char_str)
                .expect("Expect text can be encoded.");
            tokens.push(*t);
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores.
        loop {
            let mut best_score: f32 = f32::NEG_INFINITY;
            let mut best_token: usize = 0;
            let mut best_idx: usize = 0;
            let mut can_merge: bool = false;
            for i in 0..(tokens.len() - 1) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                let merged = format!("{}{}", self.vocab[tokens[i]], self.vocab[tokens[i + 1]]);
                if let Some(t) = self.vocab_map.get(&merged) {
                    if self.scores[*t] > best_score {
                        best_score = self.scores[*t];
                        best_token = *t;
                        best_idx = i;
                        can_merge = true;
                    }
                }
            }
            if !can_merge {
                break; // we couldn't find any more pairs to merge, so we're done
            }
            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_token;
            // delete token at position best_idx+1, shift the entire sequence back 1
            tokens.remove(best_idx + 1);
        }

        tokens
    }
}

#[derive(Debug, Clone, Default)]
pub struct RunState {
    // current wave of activations
    x: Vec<f32>,      // activation at current time stamp (dim,)
    xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,      // query (dim,)
    k: Vec<f32>,      // key (dim,)
    v: Vec<f32>,      // value (dim,)
    att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits
    // kv cache
    key_cache: Vec<f32>,        // (layer, seq_len, dim)
    value_cache: Vec<f32>,      // (layer, seq_len, dim)
    rng: rand::rngs::ThreadRng, // Random gen.
}

impl RunState {
    pub fn new(config: &Config) -> Self {
        Self {
            x: vec![0f32; config.dim()],
            xb: vec![0f32; config.dim()],
            xb2: vec![0f32; config.dim()],
            hb: vec![0f32; config.hidden_dim()],
            hb2: vec![0f32; config.hidden_dim()],
            q: vec![0f32; config.dim()],
            k: vec![0f32; config.dim()],
            v: vec![0f32; config.dim()],
            att: vec![0f32; config.n_heads() * config.seq_len()],
            logits: vec![0f32; config.vocab_size()],
            key_cache: vec![0f32; config.n_layers() * config.seq_len() * config.dim()],
            value_cache: vec![0f32; config.n_layers() * config.seq_len() * config.dim()],
            rng: thread_rng(),
        }
    }

    pub fn step(
        &mut self,
        token: usize,
        pos: usize,
        config: &Config,
        weights: &TransformerWeights,
    ) {
        // a few convenience variables
        let (c, w) = (config, weights);
        let (dim, hidden_dim, seq_len) = (c.dim(), c.hidden_dim(), c.seq_len());
        let head_size = c.head_size();
        let head_size_half = c.head_size() / 2;
        let n_heads = c.n_heads();

        let token_emb = slice(w.token_embedding_table(), token * dim, dim);
        self.x.copy_from_slice(token_emb);
        let freq_cis_real_row = slice(w.freq_cis_real(), pos * head_size_half, head_size_half);
        let freq_cis_imag_row = slice(w.freq_cis_imag(), pos * head_size_half, head_size_half);

        for l in 0..c.n_layers() {
            // attention rmsnorm
            rmsnorm(
                &self.x,
                slice(w.rms_att_weight(), l * dim, dim),
                &mut self.xb,
            );

            self.qkv(w, l, dim);

            self.rope_rotation(n_heads, head_size, freq_cis_real_row, freq_cis_imag_row);

            let layer_offset = l * seq_len * dim;
            self.cache_key_value(layer_offset, pos, dim);

            self.multihead_attention(n_heads, head_size, seq_len, pos, layer_offset, dim, w, l);

            self.ffn(w, l, hidden_dim, dim);
        }

        // final rmsnorm
        self.xb2.copy_from_slice(&self.x); // Temp copy x to xb2, and used for rmsnorm below.
        rmsnorm(&self.xb2, w.rms_final_weight(), &mut self.x);

        // classifier into logits
        matmul(w.wcls(), &self.x, c.vocab_size(), &mut self.logits);
    }

    fn qkv(&mut self, w: &TransformerWeights, l: usize, dim: usize) {
        // qkv matmuls for this position
        matmul(
            slice(w.wq(), l * dim * dim, dim * dim),
            &self.xb,
            dim,
            &mut self.q,
        );
        matmul(
            slice(w.wk(), l * dim * dim, dim * dim),
            &self.xb,
            dim,
            &mut self.k,
        );
        matmul(
            slice(w.wv(), l * dim * dim, dim * dim),
            &self.xb,
            dim,
            &mut self.v,
        );
    }

    /// Applies RoPE rotation to the q and k vectors for each head
    fn rope_rotation(
        &mut self,
        n_heads: usize,
        head_size: usize,
        freq_cis_real_row: &[f32],
        freq_cis_imag_row: &[f32],
    ) {
        for h in 0..n_heads {
            let q = slice_mut(&mut self.q, h * head_size, head_size);
            let k = slice_mut(&mut self.k, h * head_size, head_size);
            for i in (0..head_size).step_by(2) {
                let (q0, q1) = (q[i], q[i + 1]);
                let (k0, k1) = (k[i], k[i + 1]);
                let (fcr, fci) = (freq_cis_real_row[i / 2], freq_cis_imag_row[i / 2]);
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }
    }

    fn cache_key_value(&mut self, layer_offset: usize, pos: usize, dim: usize) {
        // save key, value at this time step (pos) to our kv cache
        let key_cache_row = slice_mut(&mut self.key_cache, layer_offset + pos * dim, dim);
        let value_cache_row = slice_mut(&mut self.value_cache, layer_offset + pos * dim, dim);
        key_cache_row.copy_from_slice(&self.k);
        value_cache_row.copy_from_slice(&self.v);
    }

    /// Multihead attention. iterate over all heads
    fn multihead_attention(
        &mut self,
        n_heads: usize,
        head_size: usize,
        seq_len: usize,
        pos: usize,
        layer_offset: usize,
        dim: usize,
        w: &TransformerWeights,
        l: usize,
    ) {
        for h in 0..n_heads {
            // get the query vector for this head
            let q = slice(&self.q, h * head_size, head_size);
            // attention scores for this head
            let att = slice_mut(&mut self.att, h * seq_len, pos + 1);
            for t in 0..pos + 1 {
                let k = slice(
                    &self.key_cache,
                    layer_offset + t * dim + h * head_size,
                    head_size,
                );
                // calculate the attention score as the dot product of q and k
                let mut score: f32 = dotprod(q, k);
                score /= (head_size as f32).sqrt();
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax_inplace(att);

            // weighted sum of the values, store back into xb.
            let xb = slice_mut(&mut self.xb, h * head_size, head_size);
            xb.fill(0.0f32); // reset 0.
            for t in 0..pos + 1 {
                let v = slice(
                    &self.value_cache,
                    layer_offset + t * dim + h * head_size,
                    head_size,
                );
                let a = att[t];
                for i in 0..head_size {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(
            slice(w.wo(), l * dim * dim, dim * dim),
            &self.xb,
            dim,
            &mut self.xb2,
        );

        // residual connection back into x
        accum(&mut self.x, &self.xb2);
    }

    // FFN calculates: self.w2(F.silu(self.w1(x)) * self.w3(x)).
    fn ffn(&mut self, w: &TransformerWeights, l: usize, hidden_dim: usize, dim: usize) {
        // ffn rmsnorm
        rmsnorm(
            &self.x,
            slice(w.rms_ffn_weight(), l * dim, dim),
            &mut self.xb,
        );

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(
            slice(w.w1(), l * hidden_dim * dim, hidden_dim * dim),
            &self.xb,
            hidden_dim,
            &mut self.hb,
        );
        matmul(
            slice(w.w3(), l * hidden_dim * dim, hidden_dim * dim),
            &self.xb,
            hidden_dim,
            &mut self.hb2,
        );

        silu_inplace(&mut self.hb);

        // elementwise multiply with w3(x)
        for i in 0..hidden_dim {
            self.hb[i] *= self.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(
            slice(w.w2(), l * dim * hidden_dim, dim * hidden_dim),
            &self.hb,
            dim,
            &mut self.xb,
        );

        // residual connection
        accum(&mut self.x, &self.xb);
    }

    pub fn sample_token(&mut self, temperature: f32) -> usize {
        if temperature == 0.0 {
            argmax(&self.logits) // greedy argmax sampling
        } else {
            // Or softmax and sampling.
            logits_to_prob(&mut self.logits, temperature);
            // we now want to sample from this distribution to get the next token
            sample(&self.logits, &mut self.rng)
        }
    }

    /// Run inference.
    pub fn run(
        &mut self,
        prompt: &str,
        temperature: f32,
        n_steps: i32,
        config: &Config,
        weights: &TransformerWeights,
        vocab: &Vocabulary,
        is_benchmark: bool,
    ) -> f32 {
        let n_steps: usize = if n_steps >= 0 && n_steps < config.seq_len {
            n_steps
        } else {
            config.seq_len
        } as usize;

        let prompt_tokens = vocab.bpe_encode(prompt);
        let mut token: usize = BOS; // 1 = BOS token in Llama-2 sentencepiece
        let mut outputs: Vec<&str> = vec!["<s>"];
        if !is_benchmark {
            print!("<s>");
        }
        let timer = Instant::now();
        for pos in 0..n_steps {
            self.step(token, pos, config, weights);

            let next = if pos < prompt_tokens.len() {
                // if we are still processing the input prompt, force the next prompt token
                prompt_tokens[pos]
            } else {
                self.sample_token(temperature)
            };

            let out = vocab.get_token_string(next, token == BOS);
            outputs.push(out);
            if !is_benchmark {
                print!("{}", out);
                let _ = stdout().flush();
            }

            token = next;
        }

        let token_per_second: f32 = (n_steps as f32) / (timer.elapsed().as_millis() as f32) * 1e3;
        // For benchmark, delay stdout tokens, since Rust print is noticeably slower than C.
        if is_benchmark {
            for s in outputs.iter() {
                print!("{}", *s);
            }
        }
        println!("\n# token/s: {}", token_per_second);

        token_per_second
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Make sure files are existed at the root directory.
    const TEST_MODEL: &str = "../stories15M.bin";
    const TEST_TOKENIZER: &str = "../tokenizer.bin";

    #[test]
    fn test_mmap() {
        use memmap2::MmapOptions;
        let file = &File::open(TEST_MODEL).unwrap();
        let mmap_result = unsafe { MmapOptions::new().map(file) };
        assert!(mmap_result.is_ok());
        let mmap = mmap_result.unwrap();
        println!("{:?}", &mmap[0..5]);

        let block = &mmap[0..100000];
        println!("{:?}", &block[500..510]);

        let block2 = &block[2350..4500];
        println!("{:?}", &block2[500..510]);
    }

    #[test]
    fn test_config() {
        let config = Config::new_from_file(TEST_MODEL);
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.dim, 288);
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.n_layers, 6);
        assert_eq!(config.n_heads, 6);
        assert_eq!(config._n_kv_heads, 6);
        assert_eq!(config.seq_len, 256);
        assert_eq!(config._vocab_size, 32000);
        assert_eq!(config.vocab_size(), 32000);
        assert_eq!(config.is_shared(), true);
    }

    #[test]
    fn test_weights() {
        let c = Config::new_from_file(TEST_MODEL).unwrap();
        let weights = TransformerWeights::new(TEST_MODEL, &c, false);
        assert!(weights.is_ok());
        let w = weights.unwrap();
        assert_eq!(w.token_embedding_table().len(), c.vocab_size() * c.dim());
        assert_eq!(w.rms_att_weight().len(), c.n_layers() * c.dim());
        assert_eq!(w.rms_ffn_weight().len(), c.n_layers() * c.dim());
        assert_eq!(w.wq().len(), c.n_layers() * c.dim() * c.dim());
        assert_eq!(w.wk().len(), c.n_layers() * c.dim() * c.dim());
        assert_eq!(w.wv().len(), c.n_layers() * c.dim() * c.dim());
        assert_eq!(w.wo().len(), c.n_layers() * c.dim() * c.dim());
        assert_eq!(w.w1().len(), c.n_layers() * c.hidden_dim() * c.dim());
        assert_eq!(w.w2().len(), c.n_layers() * c.hidden_dim() * c.dim());
        assert_eq!(w.w3().len(), c.n_layers() * c.hidden_dim() * c.dim());
        assert_eq!(w.rms_final_weight().len(), c.dim());
        let head_size_half = c.head_size() / 2;
        assert_eq!(w.freq_cis_real().len(), c.seq_len() * head_size_half);
        assert_eq!(w.freq_cis_imag().len(), c.seq_len() * head_size_half);
        assert!(w._wcls.is_none());
        assert_eq!(w._is_shared_wcls, true);
        assert_eq!(w._is_lazy, false);
        assert_eq!(w.wcls().len(), c.vocab_size() * c.dim());
    }

    #[test]
    fn test_weights_memmap() {
        let c = Config::new_from_file(TEST_MODEL).unwrap();
        let weights_mmap = TransformerWeights::new(TEST_MODEL, &c, true);
        assert!(weights_mmap.is_ok());
        let w1 = weights_mmap.unwrap();
        let w2 = TransformerWeights::new(TEST_MODEL, &c, false).unwrap();
        assert_eq!(w1.token_embedding_table(), w2.token_embedding_table());
        assert_eq!(w1.rms_att_weight(), w2.rms_att_weight());
        assert_eq!(w1.wq(), w2.wq());
        assert_eq!(w1.wk(), w2.wk());
        assert_eq!(w1.wv(), w2.wv());
        assert_eq!(w1.wo(), w2.wo());
        assert_eq!(w1.rms_ffn_weight(), w2.rms_ffn_weight());
        assert_eq!(w1.w1(), w2.w1());
        assert_eq!(w1.w2(), w2.w2());
        assert_eq!(w1.w3(), w2.w3());
        assert_eq!(w1.rms_final_weight(), w2.rms_final_weight());
        assert_eq!(w1.freq_cis_real(), w2.freq_cis_real());
        assert_eq!(w1.freq_cis_imag(), w2.freq_cis_imag());
        assert_eq!(w1.wcls(), w2.wcls());
        assert_eq!(w1._is_shared_wcls, true);
        assert_eq!(w1._is_lazy, true);
        assert_eq!(w1._wcls_range, w1._token_embedding_table_range);
    }

    #[test]
    fn test_vocab() {
        let c = Config::new_from_file(TEST_MODEL).unwrap();
        let vocab = Vocabulary::new_from_file(TEST_TOKENIZER, &c);
        assert!(vocab.is_ok());
        let vocab = vocab.unwrap();
        assert_eq!(vocab.vocab.len(), 32000);
        assert_eq!(vocab.scores.len(), 32000);
        assert_eq!(vocab._max_token_length, 27);
        assert_eq!(&vocab.vocab[0..3], vec!["<unk>", "\n<s>\n", "\n</s>\n"]); // Check 0..3.
        assert_eq!(&vocab.vocab[BOS], "\n<s>\n"); // Check BOS.

        assert_eq!(vocab.bpe_encode("hello world"), vec![12199, 3186]);
        assert_eq!(vocab.vocab_map.get("hello"), Some(&12199));
        assert_eq!(vocab.vocab_map.get(" world"), Some(&3186));
        assert_eq!(vocab.bpe_encode(""), vec![]);
    }
}
