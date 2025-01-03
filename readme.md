# fish-oxide

A port of [fishdraw](https://github.com/LingDong-/fishdraw) to rust. Credit goes to them for the original idea and code.


## Usage

### As a cli tool:
```bash
cargo install fish-oxide
fish-oxide -f svg > fish.svg
```

### As a dependency:
Add this package to your `Cargo.toml`, and this to your code:
```rust
use fish_oxide::generate_svg;

fn main() {
    println!("{}", generate_svg());
}
```

or to customize the fish:
```rust
use fish_oxide::{generate, generate_params, Format};

fn main() {
    let mut params = generate_params();
    // modify params
    println!(
        "{}",
        generate(
            Format::Svg,
            "My Seed".to_owned(),
            Some("Make a Fish".to_owned()),
            None,
            Some(20.),
            Some(params)
        )
    );
}
```
