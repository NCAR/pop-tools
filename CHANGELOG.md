# Changelog

## v2021.5.28

([full changelog](https://github.com/NCAR/pop-tools/compare/7c993d45499ffc300e22665f0160a756dc991b81...452856dd8256989eb500021d4752cfa90eaf32bc))

### New Features

- Added CFC solubility functions [#91](https://github.com/NCAR/pop-tools/pull/91) ([@matt-long](https://github.com/matt-long))
- Extend `to_xgcm_grid_dataset()` to support POP datasets with non canonical dimensions [#94](https://github.com/NCAR/pop-tools/pull/94) ([@andersy005](https://github.com/andersy005))

### Breaking Changes

- Upgrade minimum Python version to 3.7 [#79](https://github.com/NCAR/pop-tools/pull/79) ([@andersy005](https://github.com/andersy005))
- ⬆️ Upgrade dependencies and pin minimum versions [#68](https://github.com/NCAR/pop-tools/pull/68) ([@andersy005](https://github.com/andersy005))

### Bug Fixes

- Update pooch downloader: use `CESMDATAROOT` variable when available [#52](https://github.com/NCAR/pop-tools/pull/52) ([@andersy005](https://github.com/andersy005))

### Documentation

- Experiment with using xoak for indexing [#84](https://github.com/NCAR/pop-tools/pull/84) ([@dcherian](https://github.com/dcherian))

- 📚 Documentation cleanup [#81](https://github.com/NCAR/pop-tools/pull/81) ([@andersy005](https://github.com/andersy005))
- Fix Typo in docs [#80](https://github.com/NCAR/pop-tools/pull/80) ([@andersy005](https://github.com/andersy005))
- Remove sticky sidebar due to responsiveness issues on small screens [#77](https://github.com/NCAR/pop-tools/pull/77) ([@andersy005](https://github.com/andersy005))
- Add example notebook to demonstrate POP2 heat budget closure with xgcm metrics [#75](https://github.com/NCAR/pop-tools/pull/75) ([@ALDepp](https://github.com/ALDepp))
- 📚 New docs theme & Add comments [#74](https://github.com/NCAR/pop-tools/pull/74) ([@andersy005](https://github.com/andersy005))
- Use `execution_excludepatterns` to exclude long running notebooks [#76](https://github.com/NCAR/pop-tools/pull/76) ([@andersy005](https://github.com/andersy005))

### Internal Changes

- Bump pre-commit/action from v2.0.2 to v2.0.3 [#92](https://github.com/NCAR/pop-tools/pull/92) ([@dependabot](https://github.com/dependabot))
- Bump pre-commit/action from v2.0.0 to v2.0.2 [#90](https://github.com/NCAR/pop-tools/pull/90) ([@dependabot](https://github.com/dependabot))
- Bump styfle/cancel-workflow-action from 0.8.0 to 0.9.0 [#89](https://github.com/NCAR/pop-tools/pull/89) ([@dependabot](https://github.com/dependabot))
- Update CI [#78](https://github.com/NCAR/pop-tools/pull/78) ([@andersy005](https://github.com/andersy005))

### Contributors to this release

([GitHub contributors page for this release](https://github.com/NCAR/pop-tools/graphs/contributors?from=2020-12-15&to=2021-05-29&type=c))

[@ALDepp](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3AALDepp+updated%3A2020-12-15..2021-05-29&type=Issues) | [@andersy005](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Aandersy005+updated%3A2020-12-15..2021-05-29&type=Issues) | [@dcherian](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Adcherian+updated%3A2020-12-15..2021-05-29&type=Issues) | [@dependabot](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Adependabot+updated%3A2020-12-15..2021-05-29&type=Issues) | [@Eddebbar](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3AEddebbar+updated%3A2020-12-15..2021-05-29&type=Issues) | [@matt-long](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Amatt-long+updated%3A2020-12-15..2021-05-29&type=Issues) | [@mgrover1](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Amgrover1+updated%3A2020-12-15..2021-05-29&type=Issues)

## v2020.12.15

([full changelog](https://github.com/NCAR/pop-tools/compare/4aba19d40d5aec44b6032b5031f655ed3c40040e...bd1236ca615b32595c43cfa689e85fc9a112eb9f))

### Internal Changes

- ⬆️ Upgrade dependencies and pin minimum versions [#68](https://github.com/NCAR/pop-tools/pull/68) ([@andersy005](https://github.com/andersy005))

- 💚 Migrate CI from CircleCI to GHA [#67](https://github.com/NCAR/pop-tools/pull/67) ([@andersy005](https://github.com/andersy005))
- Use vectorize instead of jit in EOS [#66](https://github.com/NCAR/pop-tools/pull/66) ([@rabernat](https://github.com/rabernat))
- Update pooch downloader: use `CESMDATAROOT` variable when available [#52](https://github.com/NCAR/pop-tools/pull/52) ([@andersy005](https://github.com/andersy005))

### Contributors to this release

([GitHub contributors page for this release](https://github.com/NCAR/pop-tools/graphs/contributors?from=2020-09-14&to=2020-12-16&type=c))

[@andersy005](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Aandersy005+updated%3A2020-09-14..2020-12-16&type=Issues) | [@klindsay28](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Aklindsay28+updated%3A2020-09-14..2020-12-16&type=Issues) | [@kmpaul](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Akmpaul+updated%3A2020-09-14..2020-12-16&type=Issues) | [@matt-long](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Amatt-long+updated%3A2020-09-14..2020-12-16&type=Issues) | [@rabernat](https://github.com/search?q=repo%3ANCAR%2Fpop-tools+involves%3Arabernat+updated%3A2020-09-14..2020-12-16&type=Issues)
