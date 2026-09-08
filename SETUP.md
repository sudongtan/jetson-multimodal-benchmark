# Jetson Orin Nano — Hardware & Environment Setup

One-time setup procedures for getting the board flashed, networked, and
ready for development. Not needed for day-to-day project work — see
`CLAUDE.md` for that.

## Hardware / environment reference
- **Board**: Jetson Orin Nano Developer Kit (Super), 8GB RAM, module P3767, carrier P3768 (kit P3766)
- **Boot storage**: 64GB microSD (`/dev/mmcblk0`) — OS lives here, ~34GB free
- **Data storage**: 223GB USB SSD, mounted at `/mnt/jetson-data`, ext4, auto-mounts via `/etc/fstab`
- **JetPack**: 6.2.2 (Jetson Linux 36.5)
- **Compute stack**: CUDA 12.6.68, TensorRT 10.3.0, cuDNN
- **Access**: SSH from Ubuntu 22.04 x86_64 host; VS Code Remote-SSH for editing

---

## 1. Entering Force Recovery Mode
Needed any time the board must be reflashed via SDK Manager.

1. Power off the Jetson (unplug the DC power supply).
2. Locate the Button Header on the carrier board — a labeled pin row
   including `FC REC` and adjacent `GND` pins.
3. Bridge `FC REC` to the neighboring `GND` pin with a conductive tool
   (jumper cap, paperclip, screwdriver tip). Hold it in place.
4. **While still holding the bridge**, plug in the DC power supply.
   The board auto-powers-on the instant power is connected, so the
   bridge must already be in contact before power lands — there's very
   little margin for timing here.
5. Keep holding the bridge for ~5 seconds after power connects, then release.
6. On the host machine, confirm:
   ```bash
   lsusb
   ```
   Look for `0955:7523 NVIDIA Corp. APX`. If absent, power off and retry
   from step 1 — reseat the USB-C cable and try a firmer/more stable
   bridge contact if it keeps failing.
7. If a monitor is attached to the Jetson, use it as a faster feedback
   loop: a **blank screen** means recovery mode engaged; the NVIDIA
   splash or UEFI text means it did not, and to retry.

## 2. SDK Manager OEM Configuration choice
- If a monitor **and** keyboard are connected to the Jetson: choose
  **Runtime**. You'll complete Ubuntu's first-boot wizard on-screen
  (language, timezone, network, username, password) with full
  visibility into what's happening.
- Only choose **Pre-Config** (bake in username/password/hostname ahead
  of time, skip the on-screen wizard) if there is genuinely no monitor
  available. This path is harder to debug if something goes wrong,
  since there's no way to see the board's state afterward except over
  network — get a monitor if at all possible instead.

## 3. Flash target: use microSD, not USB SSD, for the OS
Flashing JetPack directly to a USB-attached SSD is unreliable — it
produced an unformatted EFI System Partition (ESP) in practice, which
caused UEFI to see the drive and its partitions but drop to a shell
instead of booting. Use the microSD as the flash target for the OS
(NVIDIA's documented, supported path); use the SSD only as a
post-boot data drive (see step 6).

If you ever do hit the "boots to UEFI shell, drive visible but won't
boot" symptom on a USB target:
```bash
# On the host, with the drive attached:
lsblk
sudo blkid /dev/sdX*          # find the partition labeled "esp" — no TYPE= means unformatted
sudo dd if=<esp.img from your JetPack SDK Manager download>/Linux_for_Tegra/tools/kernel_flash/images/external/esp.img \
    of=/dev/sdX<N> bs=1M status=progress
sudo sync
sudo blkid /dev/sdX<N>        # should now show TYPE="vfat"
```
Confirm the esp.img size matches the target partition size exactly
before writing, and triple-check the device letter — this command
targets a specific partition, not the whole disk.

## 4. Fixing corrupted apt package downloads
Large `.deb` downloads (commonly `libcusolver-dev-12-6`,
`nsight-compute-2024.3.1`) can arrive corrupted (`lzma error:
compressed data is corrupt`), aborting `nvidia-jetpack` install partway.
```bash
sudo apt clean
sudo apt --fix-broken install
```
This clears the corrupted cache and re-downloads just the broken
packages. Verify afterward:
```bash
nvcc --version
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

## 5. Setting the correct power mode
Mode **numbering is board-specific** — don't assume 0 is max. Check first:
```bash
cat /etc/nvpmodel.conf | grep "POWER_MODEL"
```
On this board: `0 = 15W`, `1 = 25W`, `2 = MAXN_SUPER`. Set the max mode
and lock clocks before any benchmark run:
```bash
sudo nvpmodel -m 2
sudo jetson_clocks
sudo nvpmodel -q     # confirm it reports MAXN_SUPER
```
`jetson_clocks` does **not** persist across reboots — re-run it every
session before benchmarking.

## 6. Mounting the SSD as a data drive (not boot)
```bash
# On the host, wipe and reformat as a single ext4 partition:
lsblk                                    # identify the correct device, e.g. /dev/sdb — confirm size/model before proceeding
sudo umount /media/<mounted-path>        # if auto-mounted
sudo wipefs -a /dev/sdX
sudo parted /dev/sdX --script mklabel gpt
sudo parted /dev/sdX --script mkpart primary ext4 0% 100%
sudo mkfs.ext4 /dev/sdX1 -L jetson-data

# Move the drive to the Jetson, then on the Jetson:
lsblk                                    # confirm it shows as e.g. /dev/sda1
sudo mkdir -p /mnt/jetson-data
sudo mount /dev/sda1 /mnt/jetson-data
sudo blkid /dev/sda1                     # copy the UUID
echo "UUID=<uuid> /mnt/jetson-data ext4 defaults 0 2" | sudo tee -a /etc/fstab
sudo mount -a
df -h /mnt/jetson-data                   # confirm it's mounted with expected free space

# Fix ownership so your user (not just root) can write to it:
sudo chown -R $USER:$USER /mnt/jetson-data
```

## 7. Pointing Docker at the SSD
Keeps large container images off the space-constrained microSD.
```bash
sudo systemctl stop docker
sudo mkdir -p /mnt/jetson-data/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "data-root": "/mnt/jetson-data/docker"
}
EOF
sudo systemctl start docker
docker info | grep "Docker Root Dir"     # confirm it shows /mnt/jetson-data/docker
```
Then allow running docker without sudo:
```bash
sudo usermod -aG docker $USER
# log out and back in (exit SSH session, reconnect) for this to take effect
docker run hello-world                   # should work without sudo now
```

## 8. Installing jetson-containers
```bash
cd /mnt/jetson-data
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
bash install.sh
```

## 9. Registering the NVIDIA container runtime with Docker
`nvidia-container-toolkit` is installed by JetPack, but Docker doesn't know
about it until you register it — without this step, `docker run --runtime=nvidia
...` fails with `unknown or invalid runtime name: nvidia`, and GPU-accelerated
containers (CUDA, TensorRT, etc.) can't see the GPU at all.
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -A3 Runtimes     # confirm "nvidia" is listed alongside "runc"
```
This merges an `nvidia` entry into `/etc/docker/daemon.json` alongside the
`data-root` setting from step 7 — it doesn't overwrite it. Restarting Docker
stops any currently-running containers, so do this before starting real work,
not mid-benchmark.
