"""OpenTelemetry traces and metrics (OTLP). Disabled by default. Config aligned with homiclip-backend."""

from __future__ import annotations

from fastapi import FastAPI

from app import config

# Lazy imports so OTEL deps are optional at runtime when disabled
def _setup_otel_impl(app: FastAPI) -> None:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.baggage.propagation import W3CBaggagePropagator

    set_global_textmap(
        CompositePropagator([
            TraceContextTextMapPropagator(),
            W3CBaggagePropagator(),
        ])
    )
    resource = Resource.create({
        "service.name": config.OTEL_SERVICE_NAME,
        "service.version": config.OTEL_SERVICE_VERSION,
    })
    tracer_provider = None
    meter_provider = None
    if config.OTEL_TRACES_ENABLED:
        sampler = ParentBased(root=TraceIdRatioBased(config.OTEL_TRACES_SAMPLER_ARG))
        tracer_provider = TracerProvider(resource=resource, sampler=sampler)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(tracer_provider)
    if config.OTEL_METRICS_ENABLED:
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=60_000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
    excluded_urls = "/livez,/readyz,/metrics,/health,/docs,/redoc,/openapi.json"
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        excluded_urls=excluded_urls,
    )


def setup_otel(app: FastAPI) -> None:
    """Configure OpenTelemetry traces and metrics. No-op when OTEL_ENABLED is false."""
    if not config.OTEL_ENABLED:
        return
    _setup_otel_impl(app)


def get_trace_id_hex() -> str | None:
    """Return current span trace ID as 32-char hex, or None if no valid span."""
    if not config.OTEL_TRACES_ENABLED:
        return None
    from opentelemetry import trace
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")
