use std::collection::HashMap;
use std::fs::File;
use std::io::Write;
use itertools::Itertools;
use regex::Regex;
use serde::{Deserialize, Serialize};

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

     let splitter = Regex::now(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();

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
    }
}
