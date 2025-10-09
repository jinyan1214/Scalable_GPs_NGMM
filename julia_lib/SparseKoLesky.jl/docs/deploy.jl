include("make.jl")

using Documenter: DeployConfig, DeployDecision, HTTPS
import Documenter: deploy_folder, authentication_method, authenticated_repo_url

"""
    CloudflarePages <: DeployConfig

Implement `DeployConfig` for deploying to Cloudflare pages.
"""
struct CloudflarePages <: DeployConfig
    repo::String
end

"""
    deploy_folder(
        cfg::CloudflarePages; repo, devbranch, push_preview, devurl, kwargs...
    )

Check criteria for deployment.

See:
<https://documenter.juliadocs.org/stable/man/hosting/#Documenter.deploy_folder>
"""
function deploy_folder(
    cfg::CloudflarePages; repo, devbranch, push_preview, devurl, kwargs...
)
    return DeployDecision(;
        all_ok=true,
        branch="cloudflare-pages",
        is_preview=false,
        repo,
        subfolder="",
    )
end

"""
    authentication_method(::CloudflarePages)

Set the authentication method to `HTTPS`.
"""
authentication_method(::CloudflarePages) = HTTPS

"""
    authenticated_repo_url(cfg::CloudflarePages)

Return the GitHub URL based on environmental variables.
"""
function authenticated_repo_url(cfg::CloudflarePages)
    return "https://$(ENV["GITHUB_ACTOR"]):$(ENV["GITHUB_TOKEN"])@$(cfg.repo)"
end

repo = "github.com/stephen-huan/SparseKoLesky.jl.git"

deploydocs(;
    repo,
    branch="cloudflare-pages",
    versions=nothing,
    deploy_config=CloudflarePages(repo),
)
