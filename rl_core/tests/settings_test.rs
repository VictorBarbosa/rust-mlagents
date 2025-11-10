use rl_core::settings::RootConfig;

#[test]
fn test_deserialize_single_behavior() {
    let yaml_str = r#"
        behaviors:
          TestAgent:
            trainer_type: ppo
            max_steps: 1000
            time_horizon: 64
            summary_freq: 100
            keep_checkpoints: 5
            checkpoint_interval: 500
            hyperparameters:
              beta: 0.005
              epsilon: 0.2
              lambd: 0.95
              num_epoch: 3
              learning_rate: 0.0003
              learning_rate_schedule: linear
            network_settings:
              hidden_units: 256
              num_layers: 2
              normalize: true
            reward_signals:
              extrinsic:
                gamma: 0.995
                strength: 1.0
                network_settings:
                  hidden_units: 128
                  num_layers: 1
                  normalize: true
        env_settings:
          num_envs: 8
    "#;
    let root: RootConfig = serde_yaml::from_str(yaml_str).unwrap();
    assert_eq!(root.behaviors.len(), 1);
    let b = root.behaviors.get("TestAgent").unwrap();
    assert_eq!(b.trainer_type, "ppo");
    assert_eq!(b.max_steps, 1000);
    assert_eq!(b.network_settings.as_ref().unwrap().hidden_units, Some(256));
    assert_eq!(root.env_settings.as_ref().unwrap().num_envs, Some(8));
    let rs = b.reward_signals.as_ref().unwrap().get("extrinsic").unwrap();
    assert_eq!(rs.gamma, Some(0.995));
}
