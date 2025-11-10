use rl_core::env_manager::UnityEnvManager;
use rl_core::settings::EnvSettings;

#[test]
fn test_env_manager_defaults() {
    let mgr = UnityEnvManager::from_settings(&None);
    assert_eq!(mgr.base_port, 5005);
    assert_eq!(mgr.num_envs, 1);
    assert!(mgr.env_path.is_none());
}

#[test]
fn test_env_manager_from_settings() {
    let es = EnvSettings { env_path: Some("/path/to/app".into()), base_port: Some(6000), num_envs: Some(3), num_areas: Some(1), seed: Some(42) };
    let mgr = UnityEnvManager::from_settings(&Some(es));
    assert_eq!(mgr.base_port, 6000);
    assert_eq!(mgr.num_envs, 3);
    assert_eq!(mgr.env_path.as_deref(), Some("/path/to/app"));
}

#[test]
fn test_env_manager_start_all_no_path() {
    let mgr = UnityEnvManager { env_path: None, base_port: 5005, num_envs: 2 };
    let res = mgr.start_all();
    assert!(res.is_ok());
    assert_eq!(res.unwrap().len(), 0);
}
