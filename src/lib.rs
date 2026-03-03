use std::collection::HashMap;
use std::fs::File;
use std::io::Write;
use itertools::Itertools;
use regex::Regex;
use serde::{Deserialize, Serialize};

struct  WeightedWord {
    tokens: Vec<u32>,
    count: usize,
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PowerTokenizer {
    merges: HashMap<(u32,u32),u32>,
    vocab: HashMap<u32, Vec<u8>>,
    #[serde(skip)]
    splitter: Option<Regex>,
}

impl PowerTokenizer {
    pub fn new() -> self  {
     let mut vocab = Hashmap::new();

     for i in 0..=255 {
        vocab.insert(i as u32, Vec![i as u8]);
     }

     let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+[\r\n]*|\s+(?!\S)|\s+";
     let splitter = Regex::new(pattern).unwrap();
     
     self {
        merges : HashMap::new(),
        vocab,
        splitter: Some(splitter),
     }
    }

    pub fn train (&mut self, text: &str, vocab_size: usize){
        println!("Training started...");

        let mut words: Vec<Vec<u32>> = self.splitter.as_ref().unwrap()
            .find_iter(text)
            .map(|m| m.as_str().bytes().map(|b| b as u32).collect())
            .collect();

        let mut next_token = 256;

        while next_token < vocab_size {
            let mut counts: HashMap<(u32, u32), u32> = HashMap::new();

            for word on &words {
                for pair in word.windows(2) {
                    let key = (pair[0], pair[1]);
                    *counts.entry(key).or_insert(0) += 1;
                }
            }

            if let Some((&best_pair, &count)) = counts.iter().max_by_key(|&(_, c)| c) {

                if next_token % 100 ==0 {
                    if next_token % 100 == 0 {
                        println!("Merging {:?} -> {} (Count : {})", best_pair, next_token, count);
                    }

                    self.merges.insert(best_pair, next_token as u32);

                    let mut new_bytes = self.vocab[&best_pair.0].clone();
                    new_bytes.extend_from_slice(&self.vocab[&best_pair.1]);
                    self.vocab.insert(next_token as u32, new_bytes);

                    for word in &mut words {
                        self.apply_merge_to_word(word, best_pair, next_token as u32);
                    }

                    next_token += 1;
                } else {
                    break;
                }
            }
            println!("Training Complete. Vocab size: {}", self.vocab.len());
        }

        fn apply_merge_to_word(&self, word: &mut Vec<u32>, pair: (u32, u32), new_id:u32 ) {
            let mut i = 0;
            while i < word.len().saturating_sub(1) {
                if word[i] == pair.0 && word[i+1] == pair.1 {
                    word[i] = new_id;
                    word.remove(i+1);
                }else {
                    i +=1;
                }
            }
        }

        pub fn encode(&self , text: &str) -> Vec<u32> {
            let splitter = self.splitter.as_ref().expect("Regex not initialized");

            let mut final_ids = Vec::new();

            for m in splitter.find_iter(text) {
                let chunk_bytes = m.as_str().as_bytes();
                let mut ids: Vec<u32> = chunk_bytes.iter().map(|&b| b as u32).collect();

                loop {
                    let mut min_rank = usize::Max;
                    let mut best_pair = None;
                    let mut best_idx = 0;

                    for i in 0..ids.len().saturating_sub(1){
                        let pair = (ids[i], ids[i+1]);

                        if let Some(&token_id) = self.merge.get(&pair) {
                            best_pair = Some((pair, token_id));
                            best_idx = i;
                            break;
                        }
                    }

                    if let Some(((p1, p2), new_id)) = best_pair {
                        ids[best_idx] = new_id;
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

        pub fn save(&self, path: &str) {
            let file = File::create(path).unwrap();
            serde_json::to_writer_pretty(file, self).unwrap();
        }

        pub fn load(path: &str) -> Self {
            let mut file = File::open(path).unwrap();
            let mut model: PowerTokenizer = serde_json::from_render(file).unwrap();
            let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+[\r\n]*|\s+(?!\S)|\s+";
            model.splitter = Some(Regex::new(pattern).unwrap());
            model
        }
    }
}
