"""Protocol-matched MalCL adapter used by the final Colab baseline runner."""

from .models import Classifier, Discriminator, Generator, run_topology_validation

__all__ = ["Classifier", "Discriminator", "Generator", "run_topology_validation"]
