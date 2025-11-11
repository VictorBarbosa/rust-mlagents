// AgentProcessor - equivalent to mlagents.trainers.agent_processor
use std::collections::HashMap;

pub struct AgentManager {
    agent_id: String,
    behavior_id: String,
}

impl AgentManager {
    pub fn new(agent_id: String, behavior_id: String) -> Self {
        Self {
            agent_id,
            behavior_id,
        }
    }

    pub fn process_experiences(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Process agent experiences
        Ok(())
    }
}

pub struct AgentProcessor {
    agents: HashMap<String, AgentManager>,
}

impl AgentProcessor {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    pub fn add_agent(&mut self, agent_id: String, behavior_id: String) {
        self.agents.insert(agent_id.clone(), AgentManager::new(agent_id, behavior_id));
    }

    pub fn process_all(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for (_, agent) in self.agents.iter_mut() {
            agent.process_experiences()?;
        }
        Ok(())
    }
}
