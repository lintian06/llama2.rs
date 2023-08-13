// cargo run --release -- --model_path=./stories15M.bin
// cargo run --release -- --model_path=./stories110M.bin
// cargo run --release -- --model_path=./stories15M.bin --is_benchmark
// cargo run --release -- --model_path=./stories110M.bin --is_benchmark
extern crate clap;
use clap::Parser;

mod llama2;
mod ops;
use crate::llama2::*;

/// LLAMA2 inference program
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model path.
    #[arg(short = 'p', long = "model_path", default_value_t = String::from("./stories15M.bin"))]
    model_path: String,

    /// Tokenizer path.
    #[arg(short = 't', long = "tokenizer_path", default_value_t = String::from("./tokenizer.bin"))]
    tokenizer_path: String,

    /// Temperature in search.
    #[arg(short = 'v', long = "temperature", default_value_t = 0.9)]
    temperature: f32,

    /// Number of steps in search.
    #[arg(short = 'n', long = "n_steps", default_value_t = 256)]
    n_steps: i32,

    /// Whether to use lazy model loading via mmap.
    #[arg(short = 'm', long = "is_mmap", default_value_t = false)]
    is_mmap: bool,

    /// Whether in benchmark mode.
    #[arg(short = 'b', long = "is_benchmark", default_value_t = false)]
    is_benchmark: bool,

    #[arg(short = 's', long = "prompt", default_value_t = String::from(""))]
    prompt: String,
}

/// Runs inference with repeated experiments and gets a vec of performance (tokens/s).
fn run_inference(
    prompt: &str,
    model_path: &str,
    tokenizer_path: &str,
    temperature: f32,
    n_steps: i32,
    is_mmap: bool,
    is_benchmark: bool,
    n_repeated_experiments: u32,
) -> Vec<f32> {
    let config = Config::new_from_file(model_path).unwrap();
    println!("Config: {:?}\n", config);
    let weights = TransformerWeights::new(model_path, &config, is_mmap).unwrap();
    let vocab = Vocabulary::new_from_file(tokenizer_path, &config).unwrap();

    let mut speeds = vec![];
    for _ in 0..n_repeated_experiments {
        let state = &mut RunState::new(&config);
        speeds.push(state.run(
            prompt,
            temperature,
            n_steps,
            &config,
            &weights,
            &vocab,
            is_benchmark,
        ));
    }
    speeds
}

fn main() {
    let args = Args::parse();
    println!("Args: {:?}\n", args);

    if !args.is_benchmark {
        println!("In inference mode.");
        let _speeds = run_inference(
            &args.prompt,
            &args.model_path,
            &args.tokenizer_path,
            args.temperature,
            args.n_steps,
            args.is_mmap,
            false,
            1,
        );
    } else {
        println!("In benchmark mode.");
        for is_mmap in vec![false, true] {
            let speeds = run_inference(
                &args.prompt,
                &args.model_path,
                &args.tokenizer_path,
                args.temperature,
                args.n_steps,
                is_mmap,
                true,
                10,
            );

            let mean: f32 = speeds.iter().sum::<f32>() / speeds.len() as f32;
            let var: f32 =
                speeds.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / speeds.len() as f32;
            let std = var.sqrt();
            println!("Is mmap: {}", is_mmap);
            println!("Results: {:?}", speeds);
            println!("Mean: {}, STD: {}", mean, std);
        }
    }
}
