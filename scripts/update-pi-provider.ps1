<#
把 OpenAI 兼容模型写入 pi 的 models.json。

示例：
  pwsh -NoProfile -File scripts/update-pi-provider.ps1 `
    -BaseUrl https://qwen-api.example.com `
    -ApiKey $env:QWEN_API_KEY `
    -ServiceName qwen38-colab `
    -ModelId Qwen3.8-27B-pi-Q4_K_M
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BaseUrl,
    [string]$ApiKey = 'none',
    [string]$ServiceName = 'qwen38-colab',
    [string]$ModelId = 'Qwen3.8-27B-pi-Q4_K_M',
    [string]$DisplayName = 'Qwen3.8-27B-pi · Colab A100',
    [int]$ContextWindow = 61440,
    [int]$MaxTokens = 32768,
    [string]$ModelsPath = "$env:USERPROFILE/.pi/agent/models.json"
)

$ErrorActionPreference = 'Stop'

if ($BaseUrl -notmatch '^https?://') {
    throw 'BaseUrl 需要以 http:// 或 https:// 开头'
}
if ($ServiceName -notmatch '^[a-z0-9][a-z0-9-]*$') {
    throw 'ServiceName 只能包含小写字母、数字和连字符'
}

$BaseUrl = $BaseUrl.TrimEnd('/')
if ($BaseUrl -notmatch '/v1$') {
    $BaseUrl = "$BaseUrl/v1"
}

$parent = Split-Path -Parent $ModelsPath
if (-not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
}
if (-not (Test-Path -LiteralPath $ModelsPath)) {
    [IO.File]::WriteAllText($ModelsPath, '{"providers":{}}' + "`n")
}

$model = [ordered]@{
    id               = $ModelId
    name             = $DisplayName
    contextWindow    = $ContextWindow
    maxTokens        = $MaxTokens
    reasoning        = $true
    input            = @('text')
    thinkingLevelMap = [ordered]@{
        off     = $null
        minimal = 'low'
        low     = 'low'
        medium  = 'medium'
        high    = 'xhigh'
        xhigh   = 'xhigh'
        max     = 'xhigh'
    }
}

$provider = [pscustomobject][ordered]@{
    api     = 'openai-completions'
    apiKey  = $ApiKey
    baseUrl = $BaseUrl
    models  = @([pscustomobject]$model)
}

$json = Get-Content -LiteralPath $ModelsPath -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $json.providers) {
    throw "models.json 里没有 providers 字段：$ModelsPath"
}

$json.providers | Add-Member -NotePropertyName $ServiceName -NotePropertyValue $provider -Force

$backup = "$ModelsPath.bak-$ServiceName-$((Get-Date).ToString('yyyyMMdd-HHmmss'))"
Copy-Item -LiteralPath $ModelsPath -Destination $backup
$json | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $ModelsPath -Encoding utf8

Write-Output "已写入 $ServiceName -> $BaseUrl"
Write-Output "模型：$ServiceName/$ModelId"
Write-Output "备份：$backup"
