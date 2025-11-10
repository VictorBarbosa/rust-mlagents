// use burn::backend::ndarray::NdArray;
// use rl_core::ppo::{PPOTrainer, PPOTrainerConfig};
// use rl_core::grpc::start_grpc_server; // REMOVIDO: servidor não existe mais
// use std::path::PathBuf;
// use std::fs;
//
// #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
// async fn test_ppo_with_two_grpc_envs_creates_checkpoint() {
//     type B = NdArray;
//     let device = Default::default();
//
//     // Start two dummy gRPC Unity servers
//     let (_h1, addr1, shutdown1) = start_grpc_server("127.0.0.1:0".parse().unwrap()).await.unwrap();
//     let (_h2, addr2, shutdown2) = start_grpc_server("127.0.0.1:0".parse().unwrap()).await.unwrap();
//
//     // Clean checkpoint target
//     let ckpt_dir = PathBuf::from("checkpoints");
//     let ckpt_file = ckpt_dir.join("step_1.json");
//     if ckpt_file.exists() { let _ = fs::remove_file(&ckpt_file); }
//
//     // Build trainer pointing to both env addresses
//     let mut trainer: PPOTrainer<B> = PPOTrainer::new(
//         &device,
//         PPOTrainerConfig {
//             input_size: 4,
//             action_size: 2,
//             hidden_units: 8,
//             num_layers: 1,
//             max_steps: 1,
//             checkpoint_interval: 1,
//             num_envs: 2,
//             gamma: 0.99,
//             gae_lambda: 0.95,
//             export_onnx_every_checkpoint: false,
//         },
//         vec![addr1, addr2],
//     );
//
//     trainer.train();
//
//     assert!(ckpt_file.exists());
//     let _ = fs::remove_file(ckpt_file);
//     let _ = shutdown1.send(());
//     let _ = shutdown2.send(());
// }
