# Updates .semver and _deploy/helm/values.yaml image tag after a successful build+push.
# Expects: VERSION, VERSION_TAG (e.g. v0.0.2), PROJECT_PATH.
update_image_tag_and_semver() {
    echo "${VERSION}" > "${PROJECT_PATH}/.semver"

    local values_file="_deploy/helm/values.yaml"
    local new_tag="${VERSION_TAG}"

    if [[ -f "$values_file" ]]; then
        sed -i.bak "s|tag: v[0-9.]*|tag: ${new_tag}|" "$values_file" && rm -f "${values_file}.bak"

        echo "✅ Updated image tag in values.yaml: ${new_tag}"
    else
        echo "⚠️  File not found: ${values_file}"
    fi
}
