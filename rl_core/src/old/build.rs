fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(
            &["protos/mlagents_envs/communicator_objects/unity_to_external.proto"],
            &["protos/"],
        )?;
    Ok(())
}
