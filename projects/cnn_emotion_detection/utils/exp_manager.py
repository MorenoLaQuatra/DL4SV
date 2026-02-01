import os

class ExperimentManager:
    """
    Manages experiment tracking using CometML or simple console logging.
    """
    def __init__(self, config):
        self.config = config
        self.mode = config.experiment.tracking
        self.experiment = None

        if self.mode == "comet":
            try:
                import comet_ml
                # Check for API key in config or environment variable
                api_key = getattr(config.experiment, 'api_key', os.getenv('COMET_API_KEY'))
                
                self.experiment = comet_ml.Experiment(
                    api_key=api_key,
                    project_name=config.experiment.project_name,
                    auto_output_logging="simple"
                )
                self.experiment.set_name(config.experiment.name)
                self.experiment.log_parameters(config)
                print("CometML initialized successfully.")
            except ImportError:
                print("CometML not installed. Falling back to console mode.")
                self.mode = "console"
            except Exception as e:
                print(f"Failed to initialize CometML: {e}. Falling back to console mode.")
                self.mode = "console"
        
        if self.mode == "console":
            print("Running in Console mode (no external tracking).")

    def log_metric(self, name, value, step=None, epoch=None):
        """
        Log a metric (e.g., loss, accuracy).
        """
        if self.mode == "comet" and self.experiment:
            self.experiment.log_metric(name, value, step=step, epoch=epoch)
        else:
            # Console logging is handled by the training loop, but we can add debug info here if needed
            pass

    def log_metrics(self, metrics_dict, step=None, epoch=None):
        """
        Log multiple metrics at once.
        """
        if self.mode == "comet" and self.experiment:
            self.experiment.log_metrics(metrics_dict, step=step, epoch=epoch)

    def log_parameter(self, name, value):
        """
        Log a single hyperparameter.
        """
        if self.mode == "comet" and self.experiment:
            self.experiment.log_parameter(name, value)
        else:
            print(f"Param: {name} = {value}")

    def end(self):
        """
        End the experiment.
        """
        if self.mode == "comet" and self.experiment:
            self.experiment.end()
        print("Experiment ended.")
