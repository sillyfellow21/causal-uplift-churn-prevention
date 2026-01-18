# ============================================================================
# GENERATE ALL 7 VISUALIZATIONS - ONE-CLICK EXECUTION
# ============================================================================
# This script runs all 5 Python files in sequence to generate visualizations
# ============================================================================

Write-Host "`n=====================================================================================" -ForegroundColor Green
Write-Host "🎨 GENERATING ALL 7 PROFESSIONAL VISUALIZATIONS" -ForegroundColor Green
Write-Host "=====================================================================================" -ForegroundColor Green

# Check if virtual environment is activated
if ($env:VIRTUAL_ENV) {
    Write-Host "`n✅ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "   Activating now..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# Check if packages are installed
Write-Host "`n📦 Checking required packages..." -ForegroundColor Cyan
$required = @("pandas", "numpy", "scikit-learn", "xgboost", "matplotlib", "scipy", "shap")
$missing = @()

foreach ($pkg in $required) {
    $installed = pip list 2>$null | Select-String -Pattern "^$pkg\s"
    if (-not $installed) {
        $missing += $pkg
    }
}

if ($missing.Count -gt 0) {
    Write-Host "   Missing packages: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "   Installing now..." -ForegroundColor Yellow
    pip install pandas numpy scikit-learn xgboost matplotlib scipy requests shap --quiet
    Write-Host "   ✅ Packages installed" -ForegroundColor Green
} else {
    Write-Host "   ✅ All packages installed" -ForegroundColor Green
}

# ============================================================================
# STEP 1: Preprocessing
# ============================================================================
Write-Host "`n=====================================================================================" -ForegroundColor Cyan
Write-Host "📊 STEP 1/5: Data Preprocessing" -ForegroundColor Yellow
Write-Host "=====================================================================================" -ForegroundColor Cyan
Write-Host "Running: hillstrom_analysis.py" -ForegroundColor White
Write-Host "Expected output: hillstrom_preprocessed.csv, hillstrom_modeling_ready.csv`n" -ForegroundColor Gray

$startTime = Get-Date
python hillstrom_analysis.py
$duration = (Get-Date) - $startTime
Write-Host "`n✅ Step 1 completed in $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor Green

# ============================================================================
# STEP 2: T-Learner Training
# ============================================================================
Write-Host "`n=====================================================================================" -ForegroundColor Cyan
Write-Host "🤖 STEP 2/5: T-Learner Model Training" -ForegroundColor Yellow
Write-Host "=====================================================================================" -ForegroundColor Cyan
Write-Host "Running: uplift_t_learner.py" -ForegroundColor White
Write-Host "Expected output: uplift_predictions_t_learner.csv`n" -ForegroundColor Gray

$startTime = Get-Date
python uplift_t_learner.py
$duration = (Get-Date) - $startTime
Write-Host "`n✅ Step 2 completed in $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor Green

# ============================================================================
# STEP 3: Core Evaluation (2 PNGs)
# ============================================================================
Write-Host "`n=====================================================================================" -ForegroundColor Cyan
Write-Host "📈 STEP 3/5: Core Model Evaluation" -ForegroundColor Yellow
Write-Host "=====================================================================================" -ForegroundColor Cyan
Write-Host "Running: evaluate_uplift_model.py" -ForegroundColor White
Write-Host "Generates: qini_curve_profitability.png, strategic_quadrants.png`n" -ForegroundColor Magenta

$startTime = Get-Date
python evaluate_uplift_model.py
$duration = (Get-Date) - $startTime

if (Test-Path "qini_curve_profitability.png") {
    Write-Host "`n✅ Generated: qini_curve_profitability.png" -ForegroundColor Green
}
if (Test-Path "strategic_quadrants.png") {
    Write-Host "✅ Generated: strategic_quadrants.png" -ForegroundColor Green
}
Write-Host "✅ Step 3 completed in $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor Green

# ============================================================================
# STEP 4: Meta-Learner Comparison (1 PNG) ⭐
# ============================================================================
Write-Host "`n=====================================================================================" -ForegroundColor Cyan
Write-Host "⚖️  STEP 4/5: Meta-Learner Comparison ⭐" -ForegroundColor Yellow
Write-Host "=====================================================================================" -ForegroundColor Cyan
Write-Host "Running: compare_metalearners.py" -ForegroundColor White
Write-Host "Generates: metalearner_comparison.png`n" -ForegroundColor Magenta

$startTime = Get-Date
python compare_metalearners.py
$duration = (Get-Date) - $startTime

if (Test-Path "metalearner_comparison.png") {
    Write-Host "`n✅ Generated: metalearner_comparison.png" -ForegroundColor Green
}
Write-Host "✅ Step 4 completed in $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor Green

# ============================================================================
# STEP 5: SHAP Explainability (4 PNGs) ⭐
# ============================================================================
Write-Host "`n=====================================================================================" -ForegroundColor Cyan
Write-Host "🔍 STEP 5/5: SHAP Explainability Analysis ⭐" -ForegroundColor Yellow
Write-Host "=====================================================================================" -ForegroundColor Cyan
Write-Host "Running: explainability_shap.py" -ForegroundColor White
Write-Host "Generates: shap_summary_uplift.png, shap_individual_explanations.png," -ForegroundColor Magenta
Write-Host "           shap_feature_interactions.png, shap_by_segment.png`n" -ForegroundColor Magenta

$startTime = Get-Date
python explainability_shap.py
$duration = (Get-Date) - $startTime

$shap_files = @(
    "shap_summary_uplift.png",
    "shap_individual_explanations.png",
    "shap_feature_interactions.png",
    "shap_by_segment.png"
)

foreach ($file in $shap_files) {
    if (Test-Path $file) {
        Write-Host "`n✅ Generated: $file" -ForegroundColor Green
    }
}
Write-Host "✅ Step 5 completed in $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor Green

# ============================================================================
# FINAL SUMMARY
# ============================================================================
Write-Host "`n=====================================================================================" -ForegroundColor Green
Write-Host "🎉 ALL VISUALIZATIONS GENERATED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "=====================================================================================" -ForegroundColor Green

Write-Host "`n📊 PNG Files Created:" -ForegroundColor Cyan
$pngFiles = Get-ChildItem *.png | Sort-Object Name

if ($pngFiles.Count -eq 0) {
    Write-Host "   ❌ No PNG files found!" -ForegroundColor Red
} else {
    $totalSize = 0
    foreach ($file in $pngFiles) {
        $sizeKB = [math]::Round($file.Length / 1KB, 1)
        $totalSize += $sizeKB
        Write-Host "   ✅ $($file.Name) - $sizeKB KB" -ForegroundColor Green
    }
    Write-Host "`n   Total: $($pngFiles.Count) files, $([math]::Round($totalSize/1024, 2)) MB" -ForegroundColor Yellow
}

Write-Host "`n📁 Location: $(Get-Location)" -ForegroundColor Cyan

Write-Host "`n💡 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. View PNGs: Invoke-Item *.png" -ForegroundColor White
Write-Host "   2. Check README.md for visualization descriptions" -ForegroundColor White
Write-Host "   3. Ready for GitHub upload!" -ForegroundColor White

Write-Host "`n=====================================================================================" -ForegroundColor Green
