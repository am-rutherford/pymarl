REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .async_episode_runner import AsyncEpisodeRunner
REGISTRY["async"] = AsyncEpisodeRunner

from .render_episode_runner import RenderEpisodeRunner
REGISTRY["render"] = RenderEpisodeRunner

from .timelim_episode_runner import TimeLimEpisodeRunner
REGISTRY["timelim"] = TimeLimEpisodeRunner
