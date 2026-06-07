"""Hierarchical alert type parser for AIT-ADS-A short codes.

Parses the 3-letter hierarchical short codes (e.g. W-Aut-Pam2) into structured
categories with MITRE-like attack stage mappings. Enables semantic matching
that understands that W-Aut-Pam2 (Wazuh auth failure) and W-Aut-Ssh1 (Wazuh
SSH auth) share the same attack stage (credential_access) even though their
raw short codes have zero Jaccard overlap.

The parser extracts three levels from each short code:
  - Source: A=AMiner, S=Suricata, W=Wazuh
  - Category: Acc=Access, Aut=Auth, Dns=DNS, Tls=TLS, etc.
  - Detail: the specific sub-type (Pam2, Ssh1, Hnd, etc.)

And maps each to a MITRE-like attack stage for cross-category reasoning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AlertSource(Enum):
    """IDS source that generated the alert."""
    AMINER = "A"
    SURICATA = "S"
    WAZUH = "W"
    UNKNOWN = "?"


class AttackStage(Enum):
    """MITRE ATT&CK-inspired attack stages for alert classification.

    Ordered roughly by kill-chain progression:
    recon → initial_access → execution → persistence → privilege_escalation
    → defense_evasion → credential_access → discovery → lateral_movement
    → collection → command_and_control → exfiltration → impact

    'benign' is for alerts that are typically false positives or noise.
    """
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"
    BENIGN = "benign"


@dataclass(frozen=True)
class AlertTypeInfo:
    """Parsed and enriched alert type information."""
    short_code: str
    source: AlertSource
    category: str
    detail: str
    stage: AttackStage
    description: str = ""


# ---- Category-to-stage mapping ----
# A category may map to one primary attack stage or multiple related stages.
# We use the primary stage for similarity comparison.

CATEGORY_STAGE_MAP: dict[str, AttackStage] = {
    # AMiner categories
    "Acc": AttackStage.RECONNAISSANCE,     # Access logs (new methods, characters, status codes)
    "All": AttackStage.BENIGN,             # Generic event types
    "Aud": AttackStage.DISCOVERY,          # Audit log anomaly (apparmor, login, cred, service, syscall, user)
    "Dns": AttackStage.COMMAND_AND_CONTROL, # DNS queries, entropy, IPs
    "Mon": AttackStage.IMPACT,             # CPU monitoring anomalies

    # Suricata categories
    "Dns": AttackStage.COMMAND_AND_CONTROL, # DNS queries, domain lookups
    "Flw": AttackStage.COLLECTION,         # Network flow (APT user-agent, Nmap, COVID domain)
    "Htt": AttackStage.COMMAND_AND_CONTROL, # HTTP response matching, gzip
    "Nat": AttackStage.COMMAND_AND_CONTROL, # NAT traversal (STUN)
    "Smt": AttackStage.COMMAND_AND_CONTROL, # SMTP replies
    "Tls": AttackStage.COMMAND_AND_CONTROL, # TLS handshake, certificate, record

    # Wazuh categories
    "Acc": AttackStage.RECONNAISSANCE,     # Web access (400, 500, attacks, CMS, brute force, suspicious)
    "All": AttackStage.BENIGN,             # Generic IDS/multi-source alerts
    "Aut": AttackStage.CREDENTIAL_ACCESS,  # Auth (PAM, SSH, sudo, UID changes)
    "Aud": AttackStage.DISCOVERY,          # Auditd/SELinux
    "Err": AttackStage.BENIGN,             # HTTP 403 forbidden
    "Mai": AttackStage.INITIAL_ACCESS,     # Mail (Dovecot brute force, invalid login)
    "Sys": AttackStage.EXECUTION,          # System (ClamAV, Dovecot success, auth failure)
}

# ---- Description mapping for all known short codes ----
# Maps short code → (description, primary attack stage override)
# If stage override is None, CATEGORY_STAGE_MAP is used.

ALERT_TYPE_CATALOG: dict[str, tuple[str, Optional[AttackStage]]] = {
    # === AMiner alerts ===
    "A-Acc-Chr1": ("AMiner: New characters in Apache Access referer.", AttackStage.RECONNAISSANCE),
    "A-Acc-Chr2": ("AMiner: New characters in Apache Access request.", AttackStage.RECONNAISSANCE),
    "A-Acc-Clc":  ("AMiner: Unusual occurrence frequencies of Apache Access request methods.", AttackStage.RECONNAISSANCE),
    "A-Acc-Ent1": ("AMiner: High entropy in Apache Access referer.", AttackStage.EXFILTRATION),
    "A-Acc-Ent2": ("AMiner: High entropy in Apache Access request.", AttackStage.RECONNAISSANCE),
    "A-Acc-Ent3": ("AMiner: High entropy in Apache Access user agent.", AttackStage.COMMAND_AND_CONTROL),
    "A-Acc-Frq":  ("AMiner: Unusual occurrence frequencies of Apache Access logs.", AttackStage.RECONNAISSANCE),
    "A-Acc-Val1": ("AMiner: New request method in Apache Access log.", AttackStage.RECONNAISSANCE),
    "A-Acc-Val2": ("AMiner: New status code in Apache Access log.", AttackStage.RECONNAISSANCE),
    "A-All-Evt":  ("AMiner: New event type.", AttackStage.BENIGN),
    "A-Aud-Com1": ("AMiner: New apparmor parameter combination in Audit logs.", AttackStage.DISCOVERY),
    "A-Aud-Com2": ("AMiner: New cred_acq parameter combination in Audit logs.", AttackStage.CREDENTIAL_ACCESS),
    "A-Aud-Com3": ("AMiner: New login parameter combination in Audit logs.", AttackStage.CREDENTIAL_ACCESS),
    "A-Aud-Com4": ("AMiner: New service_start parameter combination in Audit logs.", AttackStage.DISCOVERY),
    "A-Aud-Com5": ("AMiner: New syscall parameter combination in Audit logs.", AttackStage.DISCOVERY),
    "A-Aud-Com6": ("AMiner: New user_acct parameter combination in Audit logs.", AttackStage.CREDENTIAL_ACCESS),
    "A-Dns-Chr":  ("AMiner: New characters in DNS domain.", AttackStage.COMMAND_AND_CONTROL),
    "A-Dns-Clc1": ("AMiner: Unusual occurrence frequencies of DNS log events.", AttackStage.COMMAND_AND_CONTROL),
    "A-Dns-Clc2": ("AMiner: Unusual occurrence frequencies of DNS query IPs.", AttackStage.COMMAND_AND_CONTROL),
    "A-Dns-Clc3": ("AMiner: Unusual occurrence frequencies of DNS query records.", AttackStage.COMMAND_AND_CONTROL),
    "A-Dns-Ent":  ("AMiner: High entropy in DNS domain.", AttackStage.COMMAND_AND_CONTROL),
    "A-Dns-Frq":  ("AMiner: Unusual occurrence frequencies of query records in DNS logs.", AttackStage.COMMAND_AND_CONTROL),
    "A-Dns-Val1": ("AMiner: New ip address in DNS logs.", AttackStage.COMMAND_AND_CONTROL),
    "A-Dns-Val2": ("AMiner: New query record in DNS logs.", AttackStage.COMMAND_AND_CONTROL),
    "A-Mon-Avg":  ("AMiner: CPU value deviates from average in monitoring logs.", AttackStage.IMPACT),
    "A-Mon-Rng":  ("AMiner: CPU value out of expected range in monitoring logs.", AttackStage.IMPACT),

    # === Suricata alerts ===
    "S-Dns-Dom": ("Suricata: Alert - ET INFO Suspicious Domain (*.ga) in TLS SNI", AttackStage.COMMAND_AND_CONTROL),
    "S-Dns-Loo": ("Suricata: Alert - ET DNS DNS Lookup for localhost.DOMAIN.TLD", AttackStage.COMMAND_AND_CONTROL),
    "S-Dns-Qry1": ("Suricata: Alert - ET DNS Query for .to TLD", AttackStage.COMMAND_AND_CONTROL),
    "S-Dns-Qry2": ("Suricata: Alert - ET INFO DNS Query for Suspicious .ga Domain", AttackStage.COMMAND_AND_CONTROL),
    "S-Dns-Qry3": ("Suricata: Alert - ET INFO Observed DNS Query to .biz TLD", AttackStage.COMMAND_AND_CONTROL),
    "S-Dns-Qry4": ("Suricata: Alert - ET INFO Observed DNS Query to .cloud TLD", AttackStage.COMMAND_AND_CONTROL),
    "S-Dns-Uns":  ("Suricata: Alert - SURICATA DNS Unsolicited response", AttackStage.COMMAND_AND_CONTROL),
    "S-Flw-445":  ("Suricata: Alert - ET SCAN Behavioral Unusual Port 445 traffic", AttackStage.COLLECTION),
    "S-Flw-Apt":  ("Suricata: Alert - ET POLICY GNU/Linux APT User-Agent Outbound", AttackStage.COLLECTION),
    "S-Flw-Cov":  ("Suricata: Alert - ET HUNTING Suspicious Domain Request for Possible COVID-19 Dom", AttackStage.COLLECTION),
    "S-Flw-Nmp":  ("Suricata: Alert - ET SCAN Possible Nmap User-Agent Observed", AttackStage.COLLECTION),
    "S-Htt-Gzp":  ("Suricata: Alert - SURICATA HTTP gzip decompression failed", AttackStage.COMMAND_AND_CONTROL),
    "S-Htt-Mat":  ("Suricata: Alert - SURICATA HTTP unable to match response to request", AttackStage.COMMAND_AND_CONTROL),
    "S-Htt-Res":  ("Suricata: Alert - SURICATA HTTP invalid response chunk len", AttackStage.COMMAND_AND_CONTROL),
    "S-Nat-Trv":   ("Suricata: Alert - ET INFO Session Traversal Utilities for NAT (STUN Binding Req)", AttackStage.COMMAND_AND_CONTROL),
    "S-Smt-Rep":  ("Suricata: Alert - SURICATA SMTP invalid reply", AttackStage.COMMAND_AND_CONTROL),
    "S-Smt-Wel":  ("Suricata: Alert - SURICATA SMTP no server welcome message", AttackStage.COMMAND_AND_CONTROL),
    "S-Tls-Crt":  ("Suricata: Alert - SURICATA TLS certificate invalid der", AttackStage.COMMAND_AND_CONTROL),
    "S-Tls-Fai":  ("Suricata: Alert - ET INFO TLS Handshake Failure", AttackStage.COMMAND_AND_CONTROL),
    "S-Tls-Hnd":  ("Suricata: Alert - SURICATA TLS invalid handshake message", AttackStage.COMMAND_AND_CONTROL),
    "S-Tls-Rec":  ("Suricata: Alert - SURICATA TLS invalid record/traffic", AttackStage.COMMAND_AND_CONTROL),
    "S-Tls-Ssl":  ("Suricata: Alert - SURICATA TLS invalid SSLv2 header", AttackStage.COMMAND_AND_CONTROL),
    "S-Tls-Typ":  ("Suricata: Alert - SURICATA TLS invalid record type", AttackStage.COMMAND_AND_CONTROL),

    # === Wazuh alerts ===
    "W-Acc-400": ("Wazuh: Web server 400 error code.", AttackStage.RECONNAISSANCE),
    "W-Acc-500": ("Wazuh: Web server 500 error code (Internal Error).", AttackStage.RECONNAISSANCE),
    "W-Acc-Att": ("Wazuh: Common web attack.", AttackStage.INITIAL_ACCESS),
    "W-Acc-Brt": ("Wazuh: CMS (WordPress or Joomla) brute force attempt.", AttackStage.CREDENTIAL_ACCESS),
    "W-Acc-Cms": ("Wazuh: CMS (WordPress or Joomla) login attempt.", AttackStage.CREDENTIAL_ACCESS),
    "W-Acc-Sus": ("Wazuh: Suspicious URL access.", AttackStage.RECONNAISSANCE),
    "W-All-Evt": ("Wazuh: IDS event.", AttackStage.BENIGN),
    "W-All-Ids": ("Wazuh: First time this IDS alert is generated.", AttackStage.BENIGN),
    "W-All-Mul1": ("Wazuh: Multiple IDS alerts for same id.", AttackStage.BENIGN),
    "W-All-Mul2": ("Wazuh: Multiple IDS events from same source ip.", AttackStage.BENIGN),
    "W-All-Mul3": ("Wazuh: Multiple web server 400 error codes from same source ip.", AttackStage.RECONNAISSANCE),
    "W-Aud-Sel": ("Wazuh: Auditd: SELinux permission check.", AttackStage.DISCOVERY),
    "W-Aut-Pam1": ("Wazuh: PAM: Login session closed.", AttackStage.CREDENTIAL_ACCESS),
    "W-Aut-Pam2": ("Wazuh: PAM: User login failed.", AttackStage.CREDENTIAL_ACCESS),
    "W-Aut-Pam3": ("Wazuh: PAM: Multiple failed logins in a small period of time.", AttackStage.CREDENTIAL_ACCESS),
    "W-Aut-Ssh1": ("Wazuh: sshd: authentication success.", AttackStage.CREDENTIAL_ACCESS),
    "W-Aut-Ssh2": ("Wazuh: sshd: insecure connection attempt (scan).", AttackStage.RECONNAISSANCE),
    "W-Aut-Sud":  ("Wazuh: First time user executed sudo.", AttackStage.PRIVILEGE_ESCALATION),
    "W-Aut-Uid":  ("Wazuh: User successfully changed UID.", AttackStage.PRIVILEGE_ESCALATION),
    "W-Err-Fbd1": ("Wazuh: Apache: Attempt to access forbidden directory index.", AttackStage.BENIGN),
    "W-Err-Fbd2": ("Wazuh: Apache: Attempt to access forbidden file or directory.", AttackStage.BENIGN),
    "W-Mai-Brt": ("Wazuh: Dovecot brute force attack (multiple auth failures).", AttackStage.CREDENTIAL_ACCESS),
    "W-Mai-Inv": ("Wazuh: Dovecot Invalid User Login Attempt.", AttackStage.CREDENTIAL_ACCESS),
    "W-Sys-Cav": ("Wazuh: ClamAV database update", AttackStage.BENIGN),
    "W-Sys-Dov": ("Wazuh: Dovecot Authentication Success.", AttackStage.CREDENTIAL_ACCESS),
    "W-Sys-Fai": ("Wazuh: syslog: User authentication failure.", AttackStage.CREDENTIAL_ACCESS),
}


def parse_short(short_code: str) -> AlertTypeInfo:
    """Parse an AIT-ADS-A short code into structured alert type information.

    Args:
        short_code: The short code string (e.g., "W-Aut-Pam2")

    Returns:
        AlertTypeInfo with source, category, detail, stage, and description.
    """
    if not short_code or not isinstance(short_code, str):
        return AlertTypeInfo(
            short_code=str(short_code),
            source=AlertSource.UNKNOWN,
            category="Unknown",
            detail="",
            stage=AttackStage.BENIGN,
            description="Unknown alert type",
        )

    # Check catalog first for full description and stage override
    if short_code in ALERT_TYPE_CATALOG:
        desc, stage_override = ALERT_TYPE_CATALOG[short_code]
        # Parse source
        source_char = short_code[0] if short_code else "?"
        source_map = {"A": AlertSource.AMINER, "S": AlertSource.SURICATA, "W": AlertSource.WAZUH}
        source = source_map.get(source_char, AlertSource.UNKNOWN)

        # Parse category and detail
        parts = short_code.split("-", 2)
        category = parts[1] if len(parts) >= 2 else "Unknown"
        detail = parts[2] if len(parts) >= 3 else ""

        stage = stage_override if stage_override else CATEGORY_STAGE_MAP.get(category, AttackStage.BENIGN)

        return AlertTypeInfo(
            short_code=short_code,
            source=source,
            category=category,
            detail=detail,
            stage=stage,
            description=desc,
        )

    # Unknown short code — parse what we can
    parts = short_code.split("-", 2)
    source_char = parts[0] if parts else "?"
    source_map = {"A": AlertSource.AMINER, "S": AlertSource.SURICATA, "W": AlertSource.WAZUH}
    source = source_map.get(source_char, AlertSource.UNKNOWN)

    category = parts[1] if len(parts) >= 2 else "Unknown"
    detail = parts[2] if len(parts) >= 3 else ""

    stage = CATEGORY_STAGE_MAP.get(category, AttackStage.BENIGN)

    return AlertTypeInfo(
        short_code=short_code,
        source=source,
        category=category,
        detail=detail,
        stage=stage,
        description=f"Unknown alert type: {short_code}",
    )


# Pre-compute lookup for performance
_PARSED_CACHE: dict[str, AlertTypeInfo] = {}


def parse_short_cached(short_code: str) -> AlertTypeInfo:
    """Cached version of parse_short for performance in hot loops."""
    if short_code not in _PARSED_CACHE:
        _PARSED_CACHE[short_code] = parse_short(short_code)
    return _PARSED_CACHE[short_code]


def category_similarity(types_a: list[str], types_b: list[str]) -> float:
    """Compute category-level similarity between two sets of alert types.

    Instead of raw Jaccard on short codes (which gives 0 for "W-Aut-Pam2" vs "W-Aut-Ssh1"),
    this computes Jaccard similarity on the parsed category labels.

    Args:
        types_a: List of short codes from cluster A
        types_b: List of short codes from cluster B

    Returns:
        Category Jaccard similarity in [0, 1]
    """
    if not types_a or not types_b:
        return 0.0

    cats_a = set(parse_short_cached(t).category for t in types_a)
    cats_b = set(parse_short_cached(t).category for t in types_b)

    intersection = cats_a & cats_b
    union = cats_a | cats_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


def stage_similarity(types_a: list[str], types_b: list[str]) -> float:
    """Compute attack stage similarity between two sets of alert types.

    Uses the MITRE-like stage mapping to determine whether two clusters
    share attack stages (e.g., both are in credential_access stage).

    Args:
        types_a: List of short codes from cluster A
        types_b: List of short codes from cluster B

    Returns:
        Stage Jaccard similarity in [0, 1]
    """
    if not types_a or not types_b:
        return 0.0

    stages_a = set(parse_short_cached(t).stage for t in types_a)
    stages_b = set(parse_short_cached(t).stage for t in types_b)

    intersection = stages_a & stages_b
    union = stages_a | stages_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


def enriched_similarity(
    types_a: list[str],
    types_b: list[str],
    hosts_a: list[str] | None = None,
    hosts_b: list[str] | None = None,
    category_weight: float = 0.7,
) -> float:
    """Compute enriched similarity combining category, stage, and host overlap.

    The weighted combination provides more nuanced matching than raw Jaccard:
    - Category similarity captures whether two clusters share the same alert
      domain (e.g., both are authentication events)
    - Stage similarity captures whether both clusters are in the same attack
      phase (e.g., both are credential_access)
    - Host Jaccard captures whether the same hosts are involved

    The formula is:
        enriched = category_weight * category_sim + (1 - category_weight) * host_jaccard

    Stage similarity is used as a separate gate (see _enriched_semantic_match
    in the hybrid pipeline).

    Args:
        types_a: List of short codes from cluster A
        types_b: List of short codes from cluster B
        hosts_a: List of host identifiers from cluster A (optional)
        hosts_b: List of host identifiers from cluster B (optional)
        category_weight: Weight for category similarity vs host Jaccard (0-1)

    Returns:
        Weighted enrichment score in [0, 1]
    """
    cat_sim = category_similarity(types_a, types_b)

    host_sim = 0.0
    if hosts_a and hosts_b:
        set_a = set(hosts_a)
        set_b = set(hosts_b)
        if set_a and set_b:
            host_sim = len(set_a & set_b) / len(set_a | set_b)

    return category_weight * cat_sim + (1 - category_weight) * host_sim