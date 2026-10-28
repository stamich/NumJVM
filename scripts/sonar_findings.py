"""Lists NumJVM's open SonarCloud findings: ratings, Reliability and Security issues, hotspots.

Reads SonarCloud's web API, which needs no token for a public project. Used to build
docs/quality/sonar-baseline.md and, in the sonar workflow, the job summary.

Usage:
  python3 scripts/sonar_findings.py                      # main branch, Markdown on stdout
  python3 scripts/sonar_findings.py --pull-request 12    # findings of one pull request
  python3 scripts/sonar_findings.py --all                # also list maintainability issues

Environment: SONAR_HOST_URL (default https://sonarcloud.io); SONAR_TOKEN, if set, is sent
as a bearer token (needed only for private projects). Standard library only.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

RATINGS = {"1.0": "A", "2.0": "B", "3.0": "C", "4.0": "D", "5.0": "E"}
METRICS = ["reliability_rating", "security_rating", "sqale_rating", "security_review_rating", "bugs",
           "vulnerabilities", "code_smells", "security_hotspots", "coverage", "duplicated_lines_density", "ncloc"]
PAGE = 500


class Api:
    def __init__(self, host, token, scope):
        self.host = host.rstrip("/")
        self.token = token
        self.scope = scope  # extra query parameters: branch or pullRequest

    def get(self, path, **params):
        params.update(self.scope)
        url = f"{self.host}/api/{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url)
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    def paged(self, path, key, **params):
        items, page = [], 1
        while True:
            data = self.get(path, ps=PAGE, p=page, **params)
            items += data.get(key, [])
            total = data.get("paging", {}).get("total", data.get("total", 0))
            if page * PAGE >= total or not data.get(key):
                return items
            page += 1


def impacts(issue):
    return ", ".join(f"{i['softwareQuality'].title()} {i['severity'].lower()}" for i in issue.get("impacts", []))


def location(component, line):
    path = component.split(":", 1)[-1]
    return f"`{path}:{line}`" if line else f"`{path}`"


def cell(text):
    return text.replace("|", "\\|").replace("\n", " ")


def report(api, project, show_all):
    out = []
    measures = api.get("measures/component", component=project, metricKeys=",".join(METRICS))
    values = {m["metric"]: m.get("value", m.get("period", {}).get("value"))
              for m in measures.get("component", {}).get("measures", [])}

    def rating(metric):
        return RATINGS.get(values.get(metric), "–")

    out.append(f"## SonarCloud findings for `{project}`\n")
    out.append("| Reliability | Security | Maintainability | Security review | Coverage | Duplication | Lines of code |")
    out.append("|---|---|---|---|---|---|---|")
    out.append(f"| {rating('reliability_rating')} ({values.get('bugs', '–')} bugs) "
               f"| {rating('security_rating')} ({values.get('vulnerabilities', '–')} vulnerabilities) "
               f"| {rating('sqale_rating')} ({values.get('code_smells', '–')} code smells) "
               f"| {rating('security_review_rating')} ({values.get('security_hotspots', '–')} hotspots) "
               f"| {values.get('coverage', '–')} % | {values.get('duplicated_lines_density', '–')} % "
               f"| {values.get('ncloc', '–')} |\n")

    issues = api.paged("issues/search", "issues", componentKeys=project, resolved="false")
    rs = [i for i in issues if any(x["softwareQuality"] in ("RELIABILITY", "SECURITY") for x in i.get("impacts", []))]
    shown = issues if show_all else rs
    title = "Open issues" if show_all else "Open Reliability and Security issues"
    out.append(f"### {title}: {len(shown)}\n")
    if shown:
        out.append("| Type | Severity | Impact | Rule | Location | Message |")
        out.append("|---|---|---|---|---|---|")
        for i in sorted(shown, key=lambda i: (i.get("type", ""), i["component"], i.get("line", 0))):
            out.append(f"| {i.get('type', '').title()} | {i.get('severity', '').title()} | {impacts(i)} "
                       f"| `{i['rule']}` | {location(i['component'], i.get('line'))} | {cell(i['message'])} |")
        out.append("")

    hotspots = api.paged("hotspots/search", "hotspots", projectKey=project)
    out.append(f"### Security hotspots: {len(hotspots)}\n")
    if hotspots:
        out.append("| Probability | Status | Rule | Location | Message |")
        out.append("|---|---|---|---|---|")
        for h in hotspots:
            out.append(f"| {h.get('vulnerabilityProbability', '').title()} | {h.get('status', '').title()} "
                       f"| `{h.get('ruleKey', '')}` | {location(h['component'], h.get('line'))} | {cell(h['message'])} |")
        out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="stamich_NumJVM", help="SonarCloud project key")
    p.add_argument("--branch", help="analysed branch (default: the main branch)")
    p.add_argument("--pull-request", help="pull request number")
    p.add_argument("--all", action="store_true", help="list maintainability issues too")
    args = p.parse_args()
    scope = {}
    if args.pull_request:
        scope["pullRequest"] = args.pull_request
    elif args.branch:
        scope["branch"] = args.branch
    api = Api(os.environ.get("SONAR_HOST_URL", "https://sonarcloud.io"), os.environ.get("SONAR_TOKEN"), scope)
    try:
        print(report(api, args.project, args.all))
    except urllib.error.HTTPError as e:
        print(f"SonarCloud returned HTTP {e.code} for {e.url}; is the project analysed and public?", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"cannot reach SonarCloud: {e.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
