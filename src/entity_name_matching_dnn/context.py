from pathlib import Path
from typing import Any, Dict, Union

from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from pluggy import PluginManager


class ProjectContext(KedroContext):
    """A subclass of KedroContext to add Spark initialisation for the pipeline."""

    def __init__(
        self,
        package_name: str,
        project_path: Union[Path, str],
        config_loader: ConfigLoader,
        hook_manager: PluginManager,
        env: str = None,
        extra_params: Dict[str, Any] = None,
    ):
        super().__init__(
            package_name, project_path, config_loader, hook_manager, env, extra_params
        )
        