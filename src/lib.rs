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

    pub fn train (&mut self, text: &str){

    }
}
