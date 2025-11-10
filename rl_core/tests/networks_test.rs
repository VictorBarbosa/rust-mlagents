use rl_core::networks::{Actor, Critic};
use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;

#[test]
fn test_actor_network_forward_pass() {
    type B = NdArray;
    let device = Default::default();
    let batch_size = 4usize;
    let input_size = 10usize;
    let hidden_size = 32usize;
    let num_layers = 2usize;
    let output_size = 5usize; // e.g., continuous actions
    let actor: Actor<B> = Actor::new(input_size, hidden_size, num_layers, output_size, &device);
    let input = Tensor::<B, 2>::zeros([batch_size, input_size], &device);
    let output = actor.forward(input);
    assert_eq!(output.dims(), [batch_size, output_size]);
}

#[test]
fn test_critic_network_forward_pass() {
    type B = NdArray;
    let device = Default::default();
    let batch_size = 4usize;
    let input_size = 10usize;
    let hidden_size = 32usize;
    let num_layers = 2usize;
    let critic: Critic<B> = Critic::new(input_size, hidden_size, num_layers, &device);
    let input = Tensor::<B, 2>::zeros([batch_size, input_size], &device);
    let output = critic.forward(input);
    assert_eq!(output.dims(), [batch_size, 1]);
}
