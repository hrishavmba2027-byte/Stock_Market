from app.config.settings import Settings
from app.services.slack import SlackNotifier


def test_slack_noop_without_webhook(tmp_path):
    settings = Settings(
        base_dir=tmp_path,
        google_credentials=tmp_path / "creds.json",
        workflow_state_path=tmp_path / "workflow_state.json",
        logs_dir=tmp_path / "logs",
        app_log_path=tmp_path / "logs" / "app.log",
        error_log_path=tmp_path / "logs" / "error.log",
        outputs_dir=tmp_path / "outputs",
        output_dir=tmp_path / "outputs" / "main",
        model_dir=tmp_path / "outputs" / "models",
        metadata_path=tmp_path / "outputs" / "metadata.json",
        workbook_path=tmp_path / "workbook.xlsx",
        slack_webhook_url="",
    )
    settings.ensure_runtime_dirs()
    notifier = SlackNotifier(settings)
    assert notifier.notify("test", {"status": "error"}, is_failure=True) is False

