# WhisperX Helm Chart

Deploy WhisperX (speech recognition API) on Kubernetes.

## Image tags

Image tag is managed by the CI build: each successful build bumps the patch in `.semver` and updates `values.yaml` with a tag like `whisperx-v0.0.2`. Do not edit `whisperx.image.tag` by hand if you use the automated pipeline.

## Install

```bash
helm install whisperx _deploy/helm -n <namespace> -f my-values.yaml
```

## Upgrade

```bash
helm upgrade whisperx _deploy/helm -n <namespace> -f my-values.yaml
```

## Values

See [values.yaml](values.yaml). Override `whisperx.image.repository` and `whisperx.image.tag` for custom images, and set `imagePullSecrets` if using a private registry.

### OpenTelemetry (traces / metrics)

Telemetry is disabled by default. To enable it, add entries to `extraEnv` (see the example at the top of [values.yaml](values.yaml)): `OTEL_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_SERVICE_VERSION`, and optionally `OTEL_TRACES_SAMPLER_ARG`. You can enable only traces or only metrics with `OTEL_TRACES_ENABLED` / `OTEL_METRICS_ENABLED`. See the [main README](../../README.md#environment-variables) for variable details.

### Traefik

Set `whisperx.traefik.enabled: true` to create an IngressRoute (Traefik v1). Configure `whisperx.traefik.host`, `whisperx.traefik.routes` (path prefix list), `whisperx.traefik.certResolver`, and optionally `whisperx.traefik.stripPrefixes` / `whisperx.traefik.middlewares`.

### Argo CD

Set `argocd.enabled: true` to create an Argo CD `Application` in the `argocd` namespace. Configure `argocd.project`, `argocd.repoURL`, `argocd.targetRevision`, `argocd.path`, and `argocd.valueFiles` to match your repo and desired values.
