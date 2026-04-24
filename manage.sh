#!/bin/bash
# =============================================================================
# manage.sh — lifecycle helper for the RFdiffusion workshop AWS instance
# =============================================================================
# Run from your laptop. Requires AWS CLI configured (`aws configure`).
#
# One-time config:
#   export WORKSHOP_INSTANCE_ID="i-0abc123def456"
#   export WORKSHOP_KEY_PATH="$HOME/.ssh/your-key.pem"
#   export AWS_REGION="us-east-1"           # or wherever you launched
#
# (Or edit the defaults at the top of this file.)
#
# Typical commands:
#   ./manage.sh status                 Show state, type, IP, cost rate
#   ./manage.sh start                  Boot instance
#   ./manage.sh stop                   Shut down (stops billing for compute)
#   ./manage.sh resize t3.medium       Switch to cheap CPU instance for editing
#   ./manage.sh resize g5.xlarge       Switch back to GPU for workshop
#   ./manage.sh ssh                    SSH into the instance
#   ./manage.sh tunnel                 SSH tunnel + port-forward for Jupyter
#   ./manage.sh elastic-ip             One-time: attach a stable IP address
# =============================================================================

set -e

# ---- Defaults (override via env vars) ---------------------------------------
INSTANCE_ID="${WORKSHOP_INSTANCE_ID:-i-REPLACE_ME}"
KEY_PATH="${WORKSHOP_KEY_PATH:-$HOME/.ssh/your-key.pem}"
REGION="${AWS_REGION:-us-east-1}"
SSH_USER="${SSH_USER:-ubuntu}"
JUPYTER_LOCAL_PORT=8888

# ---- Helpers ----------------------------------------------------------------

usage() {
    cat <<EOF
manage.sh — lifecycle helper for the workshop AWS instance

COMMANDS
  status                  Show current state, type, IP, hourly cost
  start                   Start the instance (wait until running)
  stop                    Stop the instance (wait until stopped)
  resize <instance-type>  Change instance type (auto-stops if running)
                          Cheap:  t3.medium (\$0.04/hr, CPU only)
                                  t3.small  (\$0.02/hr, CPU only)
                          GPU:    g5.xlarge (\$1.00/hr, A10G 24GB)
                                  g4dn.xlarge (\$0.53/hr, T4 16GB)
  ssh                     SSH in (picks up current public IP)
  tunnel                  SSH tunnel for JupyterLab (local port 8888)
  ip                      Print the current public IP
  elastic-ip              Allocate + attach a stable IP (one-time, ~free)
  ami-snapshot            Create an AMI snapshot (for long-term cold storage)

CONFIG
  INSTANCE_ID = $INSTANCE_ID
  KEY_PATH    = $KEY_PATH
  REGION      = $REGION
  SSH_USER    = $SSH_USER

Override via environment variables WORKSHOP_INSTANCE_ID, WORKSHOP_KEY_PATH,
AWS_REGION, or edit the top of this script.
EOF
}

check_config() {
    if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "i-REPLACE_ME" ]; then
        echo "❌ INSTANCE_ID not set."
        echo "   Run: export WORKSHOP_INSTANCE_ID=i-xxxxxxxxx"
        echo "   or edit the default in this script."
        exit 1
    fi
}

ec2_query() {
    aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
        --query "$1" --output text 2>/dev/null
}

get_state() { ec2_query 'Reservations[0].Instances[0].State.Name'; }
get_type()  { ec2_query 'Reservations[0].Instances[0].InstanceType'; }
get_ip()    { ec2_query 'Reservations[0].Instances[0].PublicIpAddress'; }

# Rough on-demand hourly rates for common instance types (us-east-1, USD)
hourly_rate() {
    case "$1" in
        g5.xlarge)    echo "1.006" ;;
        g5.2xlarge)   echo "1.212" ;;
        g4dn.xlarge)  echo "0.526" ;;
        g4dn.2xlarge) echo "0.752" ;;
        t3.medium)    echo "0.042" ;;
        t3.small)     echo "0.021" ;;
        t3.micro)     echo "0.010" ;;
        m5.large)     echo "0.096" ;;
        c5.large)     echo "0.085" ;;
        *)            echo "?" ;;
    esac
}

# ---- Commands ---------------------------------------------------------------

cmd_status() {
    check_config
    local state type ip rate
    state=$(get_state)
    type=$(get_type)
    ip=$(get_ip)
    rate=$(hourly_rate "$type")

    printf "Instance  %s\n"  "$INSTANCE_ID"
    printf "Region    %s\n"  "$REGION"
    printf "State     %s\n"  "$state"
    printf "Type      %s\n"  "$type"
    if [ "$state" = "running" ]; then
        printf "Public IP %s\n"  "$ip"
    fi
    if [ "$rate" != "?" ]; then
        if [ "$state" = "running" ]; then
            printf "Cost      ~\$%s/hr (running)\n" "$rate"
        else
            printf "Cost      \$0/hr compute while %s (+ EBS storage only)\n" "$state"
        fi
    fi
}

cmd_start() {
    check_config
    local state
    state=$(get_state)
    if [ "$state" = "running" ]; then
        echo "Already running."
        cmd_status
        return
    fi
    echo "Starting $INSTANCE_ID..."
    aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
    echo "✅ Running."
    # Give SSH a moment to finish initializing
    sleep 5
    cmd_status
    echo ""
    echo "Next:"
    echo "  $0 ssh        # to start Jupyter on the instance"
    echo "  $0 tunnel     # (in a second terminal) to forward port 8888"
}

cmd_stop() {
    check_config
    local state
    state=$(get_state)
    if [ "$state" = "stopped" ] || [ "$state" = "stopping" ]; then
        echo "Already $state."
        return
    fi
    echo "Stopping $INSTANCE_ID..."
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
    aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
    echo "✅ Stopped. Compute billing: \$0/hr."
    echo "   Still paying: EBS storage (~\$8/month for 100GB gp3)"
}

cmd_resize() {
    check_config
    local new_type="$1"
    if [ -z "$new_type" ]; then
        echo "❌ Usage: $0 resize <instance-type>"
        echo "   Common choices:"
        echo "     g5.xlarge    GPU, workshop use      (\$1.00/hr)"
        echo "     t3.medium    CPU, cheap admin work  (\$0.04/hr)"
        exit 1
    fi

    # Architecture safety check — avoid bricking the EBS env
    case "$new_type" in
        t4g.*|a1.*|m6g.*|c6g.*|r6g.*|g5g.*)
            echo "❌ Refusing to resize to $new_type — that's ARM64."
            echo "   Your installed software is x86_64. Stay on g5/g4dn/t3/m5/c5."
            exit 1
            ;;
    esac

    local state
    state=$(get_state)
    if [ "$state" = "running" ]; then
        echo "Instance is running. Stopping first..."
        aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
        aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
        echo "✅ Stopped."
    fi

    echo "Changing type to $new_type..."
    aws ec2 modify-instance-attribute --instance-id "$INSTANCE_ID" --region "$REGION" \
        --instance-type "{\"Value\":\"$new_type\"}"

    local rate
    rate=$(hourly_rate "$new_type")
    if [ "$rate" != "?" ]; then
        printf "✅ Type set to %s (~\$%s/hr when running).\n" "$new_type" "$rate"
    else
        printf "✅ Type set to %s.\n" "$new_type"
    fi
    echo "   Run '$0 start' to boot it up."
}

cmd_ssh() {
    check_config
    local ip
    ip=$(get_ip)
    if [ -z "$ip" ] || [ "$ip" = "None" ]; then
        echo "❌ No public IP. Instance is probably stopped."
        echo "   Run '$0 start' first."
        exit 1
    fi
    echo "ssh -i $KEY_PATH $SSH_USER@$ip"
    exec ssh -i "$KEY_PATH" "$SSH_USER@$ip"
}

cmd_tunnel() {
    check_config
    local ip
    ip=$(get_ip)
    if [ -z "$ip" ] || [ "$ip" = "None" ]; then
        echo "❌ No public IP. Run '$0 start' first."
        exit 1
    fi
    echo "Tunnel: localhost:${JUPYTER_LOCAL_PORT} → ${ip}:8888"
    echo "Paste your Jupyter URL (from the instance's jupyter lab terminal) in your browser."
    echo "Ctrl-C to close."
    exec ssh -L "${JUPYTER_LOCAL_PORT}:localhost:8888" -i "$KEY_PATH" "$SSH_USER@$ip"
}

cmd_ip() {
    check_config
    get_ip
}

cmd_elastic_ip() {
    check_config
    echo "Allocating Elastic IP in $REGION..."
    local alloc_id
    alloc_id=$(aws ec2 allocate-address --region "$REGION" --query 'AllocationId' --output text)
    echo "   Allocation: $alloc_id"

    echo "Associating with $INSTANCE_ID..."
    aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$alloc_id" \
        --region "$REGION" >/dev/null

    local ip
    ip=$(get_ip)
    echo "✅ Elastic IP $ip attached."
    echo "   This IP is now stable across stop/start."
    echo "   Cost: \$0/hr while attached to a running instance."
    echo "         \$0.005/hr (~\$3.60/mo) when the instance is stopped."
    echo ""
    echo "   To avoid the small stopped-instance fee later, release the"
    echo "   Elastic IP with:"
    echo "     aws ec2 release-address --allocation-id $alloc_id --region $REGION"
}

cmd_ami_snapshot() {
    check_config
    local state
    state=$(get_state)
    if [ "$state" = "running" ]; then
        echo "Note: creating an AMI from a running instance will briefly reboot it."
        echo "      Stop the instance first if that's a problem."
    fi
    local tag
    tag="rfdiff-workshop-$(date +%Y%m%d-%H%M%S)"
    echo "Creating AMI '$tag'..."
    local ami_id
    ami_id=$(aws ec2 create-image --instance-id "$INSTANCE_ID" --region "$REGION" \
        --name "$tag" --description "Workshop instance snapshot $tag" \
        --query 'ImageId' --output text)
    echo "✅ AMI: $ami_id"
    echo "   Use this to recreate the instance from scratch later."
    echo "   Cost: ~\$0.08/GB-month for the underlying snapshot (~\$8/mo for 100GB)."
    echo "   Once snapshotted, you can safely terminate the instance + delete the EBS volume"
    echo "   to stop paying for it, and recreate later from the AMI."
}

# ---- Dispatch ---------------------------------------------------------------

case "${1:-}" in
    status)        cmd_status ;;
    start)         cmd_start ;;
    stop)          cmd_stop ;;
    resize)        cmd_resize "$2" ;;
    ssh)           cmd_ssh ;;
    tunnel)        cmd_tunnel ;;
    ip)            cmd_ip ;;
    elastic-ip)    cmd_elastic_ip ;;
    ami-snapshot)  cmd_ami_snapshot ;;
    ""|-h|--help)  usage ;;
    *)             echo "Unknown command: $1"; echo; usage; exit 1 ;;
esac
