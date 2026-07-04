# Nornickel VM Stop Request

Date: 2026-07-04

Nornickel requested that teams stop using models on their servers:

```text
ПРОСИМ ОСТАНОВИТЬ ИСПОЛЬЗОВАНИЕ МОДЕЛЕЙ
```

## Requested Action

Stop Docker containers on the Nornickel VM `team123@111.88.145.15`.

## Attempted Stop

Attempted SSH access with the provided `team123` key from `/Users/ashmelev/Downloads/team123.zip`:

```text
ssh -i <team123-private-key> team123@111.88.145.15
```

Result:

```text
ssh: connect to host 111.88.145.15 port 22: Operation timed out
```

Reachability checks from the Mac:

```text
nc 111.88.145.15 22   -> timed out
nc 111.88.145.15 8080 -> timed out
curl http://111.88.145.15:8080/workspace -> 000, timed out after ~5 s
ping 111.88.145.15 -> 100% packet loss
```

Reachability checks from `docker-srv` (`root@192.168.86.16`) also timed out on ports `22` and `8080`.

## Current Status

- The public VM UI is not reachable from our side.
- SSH is not reachable from our side.
- No Docker stop command could be executed because the VM is unreachable.
- Do not redeploy, restart, or run any workload on the Nornickel VM.
- gx10 is a local/user-owned machine and was not touched by this stop request.

## Stop Command If SSH Access Returns

If `team123@111.88.145.15` becomes reachable again, run:

```bash
ids="$(sudo docker ps -q)"
if [ -n "$ids" ]; then
  sudo docker stop $ids
  sudo docker update --restart=no $ids
fi
sudo docker ps
```

This stops all running Docker containers and disables their restart policy without deleting images, containers, or mounted workspace data.
