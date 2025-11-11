// Build script for generating gRPC code from protobuf files
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = "../../ml-agents/protobuf-definitions/proto";
    let proto_path = format!("{}/mlagents_envs/communicator_objects", proto_dir);
    
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", proto_path);
    
    // Compile all proto files - use the base directory for includes
    let protos = [
        format!("{}/unity_message.proto", proto_path),
        format!("{}/unity_input.proto", proto_path),
        format!("{}/unity_output.proto", proto_path),
        format!("{}/unity_rl_input.proto", proto_path),
        format!("{}/unity_rl_output.proto", proto_path),
        format!("{}/unity_rl_initialization_input.proto", proto_path),
        format!("{}/unity_rl_initialization_output.proto", proto_path),
        format!("{}/unity_to_external.proto", proto_path),
        format!("{}/brain_parameters.proto", proto_path),
        format!("{}/agent_info.proto", proto_path),
        format!("{}/agent_action.proto", proto_path),
        format!("{}/observation.proto", proto_path),
        format!("{}/command.proto", proto_path),
        format!("{}/space_type.proto", proto_path),
        format!("{}/header.proto", proto_path),
        format!("{}/capabilities.proto", proto_path),
        format!("{}/engine_configuration.proto", proto_path),
        format!("{}/agent_info_action_pair.proto", proto_path),
        format!("{}/demonstration_meta.proto", proto_path),
        format!("{}/custom_reset_parameters.proto", proto_path),
        format!("{}/training_analytics.proto", proto_path),
    ];
    
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(&protos, &[proto_dir])?;  // Use proto_dir as include path
    
    Ok(())
}
