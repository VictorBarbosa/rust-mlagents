use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::{backend::Backend, Tensor};
use burn::module::Module;

// Based on `SimpleActor` and `ValueNetwork` from the Python code.

#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    layers: Vec<Linear<B>>,
    action_head: Linear<B>,
}

impl<B: Backend> Actor<B> {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: usize,
        device: &B::Device,
    ) -> Self {
        let mut in_features = input_size;
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(LinearConfig::new(in_features, hidden_size).init(device));
            in_features = hidden_size;
        }
        let action_head = LinearConfig::new(in_features, output_size).init(device);
        Self { layers, action_head }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for layer in self.layers.iter() {
            x = layer.forward(x);
            x = relu(x);
        }
        self.action_head.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    layers: Vec<Linear<B>>,
    value_head: Linear<B>,
}

impl<B: Backend> Critic<B> {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        device: &B::Device,
    ) -> Self {
        let mut in_features = input_size;
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(LinearConfig::new(in_features, hidden_size).init(device));
            in_features = hidden_size;
        }
        let value_head = LinearConfig::new(in_features, 1).init(device);
        Self { layers, value_head }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for layer in self.layers.iter() {
            x = layer.forward(x);
            x = relu(x);
        }
        self.value_head.forward(x)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

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
}