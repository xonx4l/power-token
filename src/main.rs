use power_token::PowerTokenizer;

fn main() {
    let text = "Hello world! This is a test of the PowerTokenizer base system.";
    let mut tokenizer = PowerTokenizer::new();
    tokenizer.train(text, 300);

    tokenizer.save("model.json");
    println!("Model saved.");

    let loaded = PowerTokenizer::load("model.json");
    let encoded = loaded.encode("Hello world!");
    let decoded = loaded.decode(&encoded);

    println!("Ids: {:?}", encoded);
    println!("Decoded: '{}'", decoded);
}