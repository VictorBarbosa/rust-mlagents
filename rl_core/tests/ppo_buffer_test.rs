use rl_core::ppo_buffer::RolloutBuffer;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() < eps }

#[test]
fn test_gae_and_returns() {
    let mut buf = RolloutBuffer::new();
    // rewards: [1,1,1], values: [0.5,0.5,0.5], dones: [F,F,T]
    buf.push(1.0, 0.5, false);
    buf.push(1.0, 0.5, false);
    buf.push(1.0, 0.5, true);
    let gamma = 0.99f32;
    let lambda = 0.95f32;
    buf.finish_path(gamma, lambda, 0.0);

    assert_eq!(buf.advantages.len(), 3);
    assert!(approx_eq(buf.advantages[2], 0.5, 1e-5));
    assert!(approx_eq(buf.advantages[1], 0.5, 1e-5));
    assert!(approx_eq(buf.advantages[0], 1.46525, 1e-4));

    assert_eq!(buf.returns.len(), 3);
    assert!(approx_eq(buf.returns[2], 1.0, 1e-5));
    assert!(approx_eq(buf.returns[1], 1.0, 1e-5));
    assert!(approx_eq(buf.returns[0], 1.96525, 1e-4));
}
