pub mod clip;
pub mod open_clip;

pub trait Tokenizer {
    fn encode(&self, text: &str, add_sot: bool, add_eot: bool) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;

    fn start_of_text_token(&self) -> u32;
    fn end_of_text_token(&self) -> u32;
    fn padding_token(&self) -> u32;
}
