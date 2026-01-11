use bdh_model::data::acquisition::{download_gutenberg_book, GUTENBERG_IDS};
use std::fs;

fn main() {
    println!("Testing Book Acquisition...");
    let output_dir = "data/books";
    fs::create_dir_all(output_dir).unwrap();
    
    // Download first 3 books
    for &id in &GUTENBERG_IDS[0..3] {
        match download_gutenberg_book(id, output_dir) {
            Ok(path) => println!("Downloaded: {}", path),
            Err(e) => println!("Error downloading {}: {}", id, e),
        }
    }
}
