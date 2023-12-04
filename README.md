# CondaTabExpansion — "Tab" `conda` commands in PowerShell!

> *Important note*: `PSReadLine` [(here)](https://github.com/PowerShell/PSReadLine) must be installed otherwise this module won't work at all!

> *You can now download `CondaTabExpansion` from PowerShell Gallery!* [(here)](https://www.powershellgallery.com/packages/CondaTabExpansion/0.1.1)

This module provides **full support** of `conda` command completion in PowerShell (for version 5 and up supposedly). You can tab and fill commands as if you were running in `bash` or other shells.

This project is inspired by the `ChocolateyTabExpansion` script in `Chocolatey` [(here)](https://github.com/chocolatey/choco).

## Installation
> If you are running PowerShell Core 6+, you can just move on since the latest `pwsh` has already shipped with `PSReadLine`. If not, please check and install it from the link above before proceeding to next steps.

### Instal from PowerShell Gallery
   You can directly download `CondaTabExpansion` module from PowerShell Gallery[(here)](https://www.powershellgallery.com/) using the command below

```PowerShell
Install-Module CondaTabExpansion
```

#### Manual Installation (Not Recommended)
   Download or copy the scripts (`CondaTabExpansion.psm1`, `CondaTabExpansion.ps1`, and `CondaTabExpansion.psd1`) to your preferred director.

For simplicity, I will assume the script will be stored under 
`$env:USERPROFILE\Documents\PowerShell\Modules\CondaTabExpansion`.

## Enable the Module
1. Use the following command to enable the module

    ```PowerShell
    Import-Module CondaTabExpansion
    ```

2. To import this module at startup, edit your PowerShell profile and add a newline as below (it will automatically import the module at startup):

    ```powershell
    ## If download from PSGallery
    Import-Module CondaTabExpansion 

    ## If download manually
    Import-Module "$env:USERPROFILE\Documents\PowerShell\Scripts\CondaTabExpansion\CondaTabExpansion.ps1" -Force
    ```

    > If you don't know where your profile is. Just type the command below and it will show the path to your profile: 

    ```powershell 
    $PROFILE 
    ```

3. Save the profile and restart your PowerShell. The tab completion for `conda` commands should be ready to go.

## Supported Commands
| Commands        | Tab Completion        |
| -------------   |:---------------------:|
| `activate`      | Show all of your local env names or auto-complete to the closest name |
| `env`           | `'list','create','remove','update','config','export','<regex>','-h'`|
| `info`          | `-h -a --all --base -e --envs -s --system --unsafe-channels --verbose -v --quiet -q --json` |
| `install`       | `-h --revision='' -c --channel='' --use-local --override-channels --repodata-fn='' --strict-channel-priority --no-channel-priority --no-deps --only-deps --no-pin --copy -C --use-index-cache -k --insecure -offline -d --dry-run --download-only --show-channel-urls --file='' --force-reinstall --freeze-installed --update-deps -S --update-all --update-specs -m --clobber --dev --name='' -n --prefix='' -p --verbose -v --quiet -q --json` |
| `update`        | `-h -c --channel='' --use-local --override-channels --repodata-fn='' --strict-channel-priority --no-channel-priority --no-deps --only-deps --no-pin --copy -C --use-index-cache -k -insecure --offline -d --dry-run -y --yes --download-only --show-channel-urls --file='' --force-reinstall --freeze-installed --update-deps -S --update-all --update-specs --clobber --name='' -n --prefix='' -p --verbose -v --quiet -q --json` |
| `list`          | `-h --show-channel-urls -c --canonical -f --full-name --explicit --md5 -e --export --no-pip--verbose -v --quiet -q --json --revision -r --name='' -n --prefix='' -p` |
| `remove`        | `-h -c --channel='' --use-local --override-channels --repodata-fn='' -all --features ==force-remove --no-pin -C --use-index-cache -k -insecure --offline -d --dry-run -y --yes --dev --name='' -n --prefix='' -p --verbose -v --quiet -q --json` |
| `create`        | `-h --clone='' -c --channel='' --use-local --override-channels --repodata-fn='' --strict-channel-priority --no-channel-priority --no-deps --only-deps --no-pin --copy -C -k --offline -d --download-only --show-channel-urls --file=''  --no-default-packages --dev --name='' -n --prefix='' -p --verbose -v --quiet -q --json` |
| `search`        |`-h --envs -i --info --subdir='' --platform='' -c --channel='' --use-local --override-channels --repodata-fn='' -C --use-index-cache -k --insecure -offline --verbose -v --quiet -q --json` |
| `clean`         | `-a --all -i --index-cache -p --packages -t --tarballs -f --force-pkgs-dirs -c --tempfiles='' -d --dry-run -y --yes --verbose -v --quiet -q --json` |
| `config`        | `--system='' --env='' --file='' --show='' --show-sources --validate --describe='' --write-default --get='' --append --prepend --add --set --remove --remove-key --stdin --verbose -v --quiet -q --json` |
| `upgrade`       | The same as `update` |
| `uninstall`     | The same as `remove` |

## Smart Complete
| Commands | Tab Completion |
| ---      | ---            |
| `activate`      | Show all of your local env names or auto-complete to the closest name |
| `uninstall`/`remove`      | Automatically search and complete the closest name from locally installed libraries |
| `install`      | Automatically search and complete the closest library from the remote repository |
| `search`      | Automatically search and complete the closest library from the remote repository |