// use rl_core::grpc::start_grpc_server; // REMOVIDO: servidor não existe mais
use rl_core::communicator_objects::{UnityMessageProto};
use rl_core::communicator_objects::unity_to_external_proto_client::UnityToExternalProtoClient;

// #[tokio::test(flavor = "multi_thread", worker_threads = 2)] // COMENTADO: depende do servidor removido
// async fn test_grpc_exchange_echo() {
//     let (server, addr, shutdown) = start_grpc_server("127.0.0.1:0".parse().unwrap()).await.unwrap();
//     let endpoint = format!("http://{}", addr);
//     let mut client = UnityToExternalProtoClient::connect(endpoint).await.unwrap();
//
//     let msg = UnityMessageProto { header: None, unity_output: None, unity_input: None };
//     let resp = client.exchange(tonic::Request::new(msg)).await.unwrap();
//     let _ = shutdown.send(());
//     let _ = server.abort();
//     assert!(resp.get_ref().unity_output.is_none());
// }
