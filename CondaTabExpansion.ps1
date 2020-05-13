$Global:CondaTabSettings = New-Object PSObject -Property @{
    AllCommands = $false
}

# The default conda path if installed by Chocolatey
$chocoCondaPath = "$env:ChocolateyToolsLocation\Anaconda3\Library\bin\conda.bat"

# The default conda path (USER)
$usrCondaPath = "$env:USERPROFILE\Anaconda3\Library\bin\conda.bat"

# The default conda path (System)
$sysCondaPath = "$env:SystemDrive\Anaconda3\Library\bin\conda.bat"

if (Test-Path -Path $chocoCondaPath -PathType Leaf) {
    $script:conda = $chocoCondaPath
} elseif (Test-Path -Path $usrCondaPath -PathType Leaf) {
    $script:conda = $usrCondaPath
} elseif (Test-Path -Path $sysCondaPath -PathType Leaf) {
    $script:conda = $sysCondaPath
} else {
    $script:conda = conda.bat   # Just use the conda.bat in PATH ... if can find one
}


function script:condaCmdOperations($commands, $command, $filter, $currentArguments) {
    $currentOptions = @('zzzz')
    if ($null -ne $currentArguments -and $currentArguments.Trim() -ne '') { $currentOptions = $currentArguments.Trim() -split ' ' }

    $commands.$command.Replace("  "," ") -split ' ' |
        Where-Object { $_ -notmatch "^(?:$($currentOptions -join '|' -replace "=", "\="))(?:\S*)\s?$" } |
        Where-Object { $_ -like "$filter*" }
}

$script:someCommands = @('activate','env','info','install','update','list','remove','create','search','clean','config','init','package','run','uninstall','upgrade','-h','--help','-V','--version')

# ensure these all have a space to start, or they will cause issues
$outputcommands = " --verbose -v --quiet -q --json"

$namecommands = " --revision -r"

$envcommands = " --name='' -n --prefix='' -p"

$commandOptions = @{
    clean = "-a --all -i --index-cache -p --packages -t --tarballs -f --force-pkgs-dirs -c --tempfiles='' -d --dry-run -y --yes" + $outputcommands

    config = "--system='' --env='' --file='' --show='' --show-sources --validate --describe='' --write-default --get='' --append --prepend --add --set --remove --remove-key --stdin" + $outputcommands

    create = "-h --clone='' -c --channel='' --use-local --override-channels --repodata-fn='' --strict-channel-priority --no-channel-priority --no-deps --only-deps --no-pin --copy -C -k --offline -d --download-only --show-channel-urls --file=''  --no-default-packages --dev" + $envcommands + $outputcommands

    info = "-h -a --all --base -e --envs -s --system --unsafe-channels" + $outputcommands

    install = "-h --revision='' -c --channel='' --use-local --override-channels --repodata-fn='' --strict-channel-priority --no-channel-priority --no-deps --only-deps --no-pin --copy -C --use-index-cache -k --insecure -offline -d --dry-run --download-only --show-channel-urls --file='' --force-reinstall --freeze-installed --update-deps -S --update-all --update-specs -m --clobber --dev" + $envcommands + $outputcommands

    list = "-h --show-channel-urls -c --canonical -f --full-name --explicit --md5 -e --export --no-pip" + $envcommands + $outputcommands + $namecommands

    package = "-h -w --which='' -r --reset -u --untracked --pkg-name='' --pkg-version='' --pkg-build=''" + $envcommands

    remove = "-h -c --channel='' --use-local --override-channels --repodata-fn='' -all --features ==force-remove --no-pin -C --use-index-cache -k -insecure --offline -d --dry-run -y --yes --dev" + $envcommands + $outputcommands

    uninstall = "-h -c --channel='' --use-local --override-channels --repodata-fn='' -all --features ==force-remove --no-pin -C --use-index-cache -k -insecure --offline -d --dry-run -y --yes --dev" + $envcommands + $outputcommands

    search = "-h --envs -i --info --subdir='' --platform='' -c --channel='' --use-local --override-channels --repodata-fn='' -C --use-index-cache -k --insecure -offline" + $outputcommands

    update = "-h -c --channel='' --use-local --override-channels --repodata-fn='' --strict-channel-priority --no-channel-priority --no-deps --only-deps --no-pin --copy -C --use-index-cache -k -insecure --offline -d --dry-run -y --yes --download-only --show-channel-urls --file='' --force-reinstall --freeze-installed --update-deps -S --update-all --update-specs --clobber" + $envcommands + $outputcommands

    upgrade = "-h -c --channel='' --use-local --override-channels --repodata-fn='' --strict-channel-priority --no-channel-priority --no-deps --only-deps --no-pin --copy -C --use-index-cache -k -insecure --offline -d --dry-run -y --yes --download-only --show-channel-urls --file='' --force-reinstall --freeze-installed --update-deps -S --update-all --update-specs --clobber" + $envcommands + $outputcommands
}

function script:condaCommands($filter) {
    $cmdList = @()
    if (-not $global:CondaTabSettings.AllCommands) {
        $cmdList += $someCommands -like "$filter*"
    } else {
        $cmdList += (& $script:conda -h) |
            Where-Object { $_ -match '^  \S.*' } |
            ForEach-Object { $_.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) } |
            Where-Object { $_ -like "$filter*" }
    }

    $cmdList #| sort
}

function script:condaLocalEnvs($filter) {
    if ($null -ne $filter -and $filter.StartsWith(".")) { return; } #file search
    @(& $script:conda env list $filter) | ForEach-Object{ 
        if ($_ -match '^\w') { $_.Split(' ')[0] } 
    } | Get-Unique
}

function script:condaLocalPackages($filter) {
    if ($null -ne $filter -and $filter.StartsWith(".")) { return; } #file search
    @(& $script:conda list $filter) | ForEach-Object{ 
        if ($_ -match '^\w') { $_.Split(' ')[0] } 
    } | Get-Unique
}

function script:condaRemotePackages($filter) {
    if ($null -ne $filter -and $filter.StartsWith(".")) { return; } #file search
    @(& $script:conda search $filter) | 
    ForEach-Object{ 
        if (-Not ($_ -match '^Loading channels: ') -and 
                 ($_ -match '^\w') -and 
            -Not ($_ -match 'No match found for:')) { 
            $_.Split(' ')[0] 
        } 
    } | Get-Unique
}

function Get-AliasPattern($exe) {
    $aliases = @($exe) + @(Get-Alias | Where-Object { $_.Definition -eq $exe } | Select-Object -Exp Name)
  
    "($($aliases -join '|'))"
}

function CondaTabExpansion($lastBlock) {
    switch -regex ($lastBlock -replace "^$(Get-AliasPattern conda) ","") {

        # Handles uninstall/remove package names
        "^(uninstall|remove)\s+(?<package>[^\.][^-\s]*)$" {
            condaLocalPackages $matches['package']
        }   

        # Handles install package names
        "^(install)\s+(?<package>[^\.][^-\s]+)$" {
            condaRemotePackages $matches['package']
        }

        # Handles search first tab
        "^(search)\s+(?<package>[^\.][^-\s]+)$" {
            condaRemotePackages $matches['package']
        }

        # Handles general activate
        "^activate(|\s+)$" {
            condaLocalEnvs 
        }

        # Handles activate env names
        "^activate\s+(?<env>[^\.][^-\s]+)$" {
            condaLocalEnvs | Where-Object { $_ -like "$($matches['env'])*" }
        }

        # Handles info first tab
        "^info\s+(?<subcommand>[^-\s]*)$" {
            @('-h') | Where-Object { $_ -like "$($matches['subcommand'])*" }
        }

        # Handles create first tab
        "^create\s+(?<subcommand>[^-\s]*)$" {
            @('<package_spec>','-h') | Where-Object { $_ -like "$($matches['subcommand'])*" }
        }

        # Handles update/upgrade first tab
        "^(update|upgrade)\s+(?<subcommand>[^-\s]*)$" {
            @('<package_spec>','-h') | Where-Object { $_ -like "$($matches['subcommand'])*" }
        }

        # Handles list first tab
        "^list\s+(?<subcommand>[^-\s]*)$" {
            @('<package_spec>','-h') | Where-Object { $_ -like "$($matches['subcommand'])*" }
        }

        # Handles clean first tab
        "^clean\s+(?<subcommand>[^-\s]*)$" {
            @('-h') | Where-Object { $_ -like "$($matches['subcommand'])*" }
        }

        # Handles config first tab
        "^config\s+(?<subcommand>[^-\s]*)$" {
            @('-h') | Where-Object { $_ -like "$($matches['subcommand'])*" }
        }

        # Handles init first tab
        "^init\s+(?<subcommand>[^-\s]*)$" {
            @('<package_spec>','-h') | Where-Object { $_ -like "$($matches['subcommand'])*" }
        }

        # Handles env first tab
        "^env\s+(?<subcommand>[^-\s]*)$" {
            @('list','create','remove','update','config','export','<regex>','-h') | Where-Object { $_ -like "$($matches['subcommand'])*" }
        }     

        # Handles more options after others
        "^(?<cmd>$($commandOptions.Keys -join '|'))(?<currentArguments>.*)\s+(?<op>\S*)$" {
            condaCmdOperations $commandOptions $matches['cmd'] $matches['op'] $matches  ['currentArguments']
        }
  
        # Handles conda <cmd> <op>
        "^(?<cmd>$($commandOptions.Keys -join '|'))\s+(?<op>\S*)$" {
            condaCmdOperations $commandOptions $matches['cmd'] $matches['op']
        }
  
        # Handles conda <cmd>
        "^(?<cmd>\S*)$" {
            if ($_ -ne "activate") {
                condaCommands $matches['cmd']
            }
        }
    }
}

$PowerTab_RegisterTabExpansion = if (Get-Module -Name powertabl) { 
    Get-Command Register-TabExpansion -Module powertab -ErrorAction SilentlyContinue
}

if ($PowerTab_RegisterTabExpansion) {
    & $PowerTab_RegisterTabExpansion "conda" -Type Command {
        param($Context, [ref]$TabExpansionHasOutput, [ref]$QuoteSpaces)
        $line = $Context.Line
        $lastBlock = [System.Text.RegularExpressions.Regex]::Split($line, '[|;]')[-1].TrimStart()
        $TabExpansionHasOutput.Value = $true
        CondaTabExpansion $lastBlock
    }

    return
}

if (Test-Path Function:\TabExpansion) {
    Rename-Item Function:\TabExpansion sysTabExpansionBackup
}

function TabExpansion($line, $lastWord) {
    $lastBlock = [System.Text.RegularExpressions.Regex]::Split($line, '[|;]')[-1].TrimStart()

    switch -regex ($lastBlock) {
        # Execute Conda tab completion for all conda-related commands
        "^$(Get-AliasPattern conda) (.*)" { CondaTabExpansion $lastBlock }

        # Fall back on exisitng tab expansion
        default { if (Test-Path Function:\sysTabExpansionBackup) {
            sysTabExpansionBackup $line $lastWord
        } }
    }
}
