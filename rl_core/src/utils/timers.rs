// Timers - equivalent to mlagents_envs.timers
use std::time::{Instant, Duration};
use std::collections::HashMap;

pub struct Timer {
    start: Instant,
    duration: Duration,
    name: String,
}

impl Timer {
    pub fn new(name: String) -> Self {
        Self {
            start: Instant::now(),
            duration: Duration::ZERO,
            name,
        }
    }

    pub fn stop(&mut self) {
        self.duration = self.start.elapsed();
    }

    pub fn elapsed(&self) -> Duration {
        self.duration
    }
}

pub struct HierarchicalTimer {
    timers: HashMap<String, Timer>,
    current_timer: Option<String>,
}

impl HierarchicalTimer {
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            current_timer: None,
        }
    }

    pub fn start(&mut self, name: String) {
        let timer = Timer::new(name.clone());
        self.timers.insert(name.clone(), timer);
        self.current_timer = Some(name);
    }

    pub fn stop(&mut self) {
        if let Some(name) = &self.current_timer {
            if let Some(timer) = self.timers.get_mut(name) {
                timer.stop();
            }
        }
        self.current_timer = None;
    }

    pub fn get_elapsed(&self, name: &str) -> Option<Duration> {
        self.timers.get(name).map(|t| t.elapsed())
    }
}

thread_local! {
    static TIMER: std::cell::RefCell<HierarchicalTimer> = std::cell::RefCell::new(HierarchicalTimer::new());
}

pub fn hierarchical_timer_start(name: &str) {
    TIMER.with(|t| t.borrow_mut().start(name.to_string()));
}

pub fn hierarchical_timer_stop() {
    TIMER.with(|t| t.borrow_mut().stop());
}

pub fn get_timer_elapsed(name: &str) -> Option<Duration> {
    TIMER.with(|t| t.borrow().get_elapsed(name))
}
