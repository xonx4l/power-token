use std::collections::{HashMap, HashSet};
use std::fs::File;
use rayon::prelude::*; 
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
struct WeightedWord {
    tokens: Vec<u32>,
    count: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PowerTokenizer {
    merges: HashMap<(u32, u32), u32>,
    vocab: HashMap<u32, Vec<u8>>,
    special_tokens: HashMap<String, u32>, 
    #[serde(skip)]
    splitter: Option<Regex>,
}

impl PowerTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();

        for i in 0..=255 {
            vocab.insert(i as u32, vec![i as u8]);
        }

        let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+[\r\n]*|\s+(?!\S)|\s+";
        let splitter = Regex::new(pattern).unwrap();

        Self {
            merges: HashMap::new(),
            vocab,
            special_tokens: HashMap::new(),
            splitter: Some(splitter),
        }
    }

    pub fn add_special_token(&mut self, token: &str) {
        let new_id = (256 + self.merges.len() + self.special_tokens.len()) as u32;
        self.special_tokens.insert(token.to_string(), new_id);
        self.vocab.insert(new_id, token.as_bytes().to_vec());
    }

    pub fn train(&mut self, text: &str, vocab_size: usize) {
        println!("Phase 1: Compressing text (Weighted Deduplication)...");
        let splitter = self.splitter.as_ref().unwrap();

        let chunks: Vec<&str> = splitter.find_iter(text).map(|m| m.as_str()).collect();
        
        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        for chunk in chunks {
            *word_counts.entry(chunk).or_insert(0) += 1;
        }

        let mut words: Vec<WeightedWord> = word_counts.into_iter().map(|(word_str, count)| {
            WeightedWord {
                tokens: word_str.bytes().map(|b| b as u32).collect(),
                count,
            }
        }).collect();

        let mut next_token = 256 + self.special_tokens.len() as u32;

        println!("Phase 2: BPE Merging...");
        while (next_token as usize) < vocab_size {
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();

            for word in &words {
                if word.tokens.len() < 2 { continue; }
                for window in word.tokens.windows(2) {
                    let key = (window[0], window[1]);
                    *counts.entry(key).or_insert(0) += word.count; 
                }
            }

            if let Some((&best_pair, &count)) = counts.iter().max_by_key(|&(_, c)| c) {
                
                if next_token % 100 == 0 {
                    println!("Merging {:?} -> {} (Occurrences: {})", best_pair, next_token, count);
                }

                self.merges.insert(best_pair, next_token);

                let mut new_bytes = self.vocab[&best_pair.0].clone();
                new_bytes.extend_from_slice(&self.vocab[&best_pair.1]);
                self.vocab.insert(next_token, new_bytes);

                for word in &mut words {
                    self.apply_merge_to_word(word, best_pair, next_token);
                }

                next_token += 1;
            } else {
                break; 
            }
        }
        println!("Training Complete. Final Vocab size: {}", self.vocab.len());
    }

    fn apply_merge_to_word(&self, word: &mut WeightedWord, pair: (u32, u32), new_id: u32) {
        let mut i = 0;
        while i < word.tokens.len().saturating_sub(1) {
            if word.tokens[i] == pair.0 && word.tokens[i + 1] == pair.1 {
                word.tokens[i] = new_id;
                word.tokens.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let splitter = self.splitter.as_ref().expect("Regex not initialized");
        let mut final_ids = Vec::new();

        for m in splitter.find_iter(text) {
            let chunk_bytes = m.as_str().as_bytes();
            let mut ids: Vec<u32> = chunk_bytes.iter().map(|&b| b as u32).collect();

            loop {
                let mut best_pair = None;
                let mut best_idx = 0;
                let mut min_token_id = u32::MAX;

                for i in 0..ids.len().saturating_sub(1) {
                    let pair = (ids[i], ids[i + 1]);

                    if let Some(&token_id) = self.merges.get(&pair) {
                        if token_id < min_token_id {
                            best_pair = Some(pair);
                            best_idx = i;
                            min_token_id = token_id;
                        }
                    }
                }

                if let Some(_) = best_pair {
                    ids[best_idx] = min_token_id;
                    ids.remove(best_idx + 1);
                } else {
                    break;
                }
            }
            final_ids.extend(ids);
        }
        final_ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(b) = self.vocab.get(&id) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn decode_stream(&self, token_id: u32, buffer: &mut Vec<u8>) -> String {
        if let Some(bytes) = self.vocab.get(&token_id){
            buffer.extend_from_slice(bytes);
        } else {
            return String::new();
        }
    }

    match std::str::from_utf8(buffer) {
        Ok(Valid_str) => {
            let result = valid_str.to_string();
            buffer.clear();
            result
        }
        Err(e) => {
            let valid_up_to = e.valid_up_to();

            if e.error_len().is_none() {
                let valid_text = String::from_utf8_lossy(&buffer[..valid_up_to]).to_string();
                buffer.drain(..valid_up_to);
                valid_text
            }else {
                let error_len = e.error_len().unwrap();
                let valid_text = String::from_utf8_lossy(&buffer[..valid_up_to]).to_string;
                let result = format!("{}", valid_text);

                buffer.drain(..valid_up_to + error_len);
                result
            }
        }
    }
    
    pub fn save(&self, path: &str) {
        let file = File::create(path).unwrap();
        serde_json::to_writer_pretty(file, self).unwrap();
        println!("Tokenizer saved to {}", path);
    }

    pub fn load(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let mut model: PowerTokenizer = serde_json::from_reader(file).unwrap();
        let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+[\r\n]*|\s+(?!\S)|\s+";
        model.splitter = Some(Regex::new(pattern).unwrap());
        model
    }
}