# PowerShell script to add Tesseract to PATH

# Potential Tesseract installation paths
$possiblePaths = @(
    "C:\Program Files\Tesseract-OCR",
    "C:\Program Files (x86)\Tesseract-OCR"
)

# Find the first existing path
$tesseractPath = $possiblePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($tesseractPath) {
    # Get current PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

    # Check if path already exists
    if ($currentPath -notlike "*$tesseractPath*") {
        # Add new path
        $newPath = $currentPath + ";$tesseractPath"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
        
        Write-Host "Tesseract path added successfully: $tesseractPath"
        Write-Host "Please restart your command prompt/IDE for changes to take effect."
    }
    else {
        Write-Host "Tesseract path already exists in PATH."
    }

    # Verify Tesseract installation
    $tesseractExe = Join-Path $tesseractPath "tesseract.exe"
    if (Test-Path $tesseractExe) {
        & $tesseractExe --version
    }
}
else {
    Write-Host "Tesseract installation directory not found. Please install Tesseract first."
}
