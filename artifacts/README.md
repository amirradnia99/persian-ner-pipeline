# Artifacts & Integrity

Large artifacts are distributed via GitHub Releases.

## Release
- Tag: v1.0.0-silver
- Asset: PW-NER.rar

## SHA-256 (Integrity Check)
The SHA-256 checksum for the release asset is stored in:
- `checksums_sha256.txt`

To verify after download (PowerShell):
```powershell
Get-FileHash .\PW-NER.rar -Algorithm SHA256
