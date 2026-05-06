//! Bundled simple formats: codecs and containers that are small and standard
//! enough to share one crate. Anything larger gets its own crate.

pub mod pcm;
pub mod slin;
pub mod wav;
pub mod y4m;

use oxideav_core::CodecRegistry;
use oxideav_core::ContainerRegistry;
use oxideav_core::RuntimeContext;

/// Register every codec provided by `oxideav-basic` in a [`CodecRegistry`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    pcm::register(reg);
}

/// Register every container provided by `oxideav-basic` in a [`ContainerRegistry`].
pub fn register_containers(reg: &mut ContainerRegistry) {
    wav::register(reg);
    slin::register(reg);
    y4m::register(reg);
}

/// Unified entry point: install every codec and container provided by
/// `oxideav-basic` into a [`RuntimeContext`].
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

oxideav_core::register!("basic", register);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_via_runtime_context_installs_factories() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        assert!(
            ctx.codecs.decoder_ids().next().is_some(),
            "register(ctx) should install codec decoder factories"
        );
        assert!(
            ctx.containers.demuxer_names().next().is_some(),
            "register(ctx) should install container demuxer factories"
        );
    }
}
