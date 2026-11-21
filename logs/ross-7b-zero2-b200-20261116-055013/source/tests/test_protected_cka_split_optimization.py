"""Filesystem-only regression tests for invocation-scoped split path caching."""

import errno
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_protected_cka_config import CONFIG, FakeDataset, args


class ImagePathResolverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.images = self.base / "images"
        self.images.mkdir()

    def assert_matches_legacy(self, image, folder=None):
        record = {"image": image}
        folder = str(self.images if folder is None else folder)
        try:
            expected = CONFIG._canonical_image(record, folder)
        except Exception as error:
            with self.assertRaises(type(error)):
                CONFIG._ImagePathResolver()(record, folder)
        else:
            self.assertEqual(CONFIG._ImagePathResolver()(record, folder), expected)

    def test_normal_absolute_parent_and_leaf_symlinks_match_legacy(self):
        real = self.images / "real"
        real.mkdir()
        (real / "nested").mkdir()
        (real / "image.jpg").touch()
        (real / "nested" / "other.jpg").touch()
        (self.images / "parent-alias").symlink_to(real, target_is_directory=True)
        (self.images / "leaf-alias.jpg").symlink_to("real/image.jpg")
        (self.images / "dangling.jpg").symlink_to("missing/target.jpg")
        (self.images / "dangling-parent").symlink_to("missing-directory", target_is_directory=True)
        paths = (
            "real/image.jpg", "./real/image.jpg", "parent-alias/image.jpg",
            "leaf-alias.jpg", "dangling.jpg", "dangling-parent/image.jpg",
            "real/missing.jpg", "missing-parent/missing.jpg", ".", "/",
            str(real / "image.jpg"), str(self.images / "leaf-alias.jpg"),
        )
        for image in paths:
            with self.subTest(image=image):
                self.assert_matches_legacy(image)
        # An absolute image ignores even a nonexistent image_folder.
        self.assert_matches_legacy(str(real / "image.jpg"), self.base / "not-a-folder")

    def test_dotdot_uses_filesystem_not_lexical_symlink_parent(self):
        real = self.images / "real"
        (real / "nested").mkdir(parents=True)
        (real / "target.jpg").touch()
        (self.images / "alias").symlink_to(real / "nested", target_is_directory=True)
        for image in ("alias/../target.jpg", "real/nested/../target.jpg",
                      "missing/../real/target.jpg", "../images/real/target.jpg"):
            with self.subTest(image=image):
                self.assert_matches_legacy(image)
        resolved = CONFIG._ImagePathResolver()({"image": "alias/../target.jpg"}, str(self.images))
        self.assertEqual(resolved, str(real / "target.jpg"))

    def test_symlink_loops_preserve_legacy_exceptions(self):
        (self.images / "loop-a").symlink_to("loop-b")
        (self.images / "loop-b").symlink_to("loop-a")
        for image in ("loop-a", "loop-b", "loop-a/child.jpg"):
            with self.subTest(image=image):
                self.assert_matches_legacy(image)

    def test_leaf_lstat_error_falls_back_to_original_full_resolve(self):
        leaf = self.images / "image.jpg"
        leaf.touch()
        original_lstat, original_resolve = os.lstat, Path.resolve
        for error_number in (errno.ENOENT, errno.EACCES, errno.EIO):
            with self.subTest(error=error_number):
                resolved = []

                def failing_lstat(path, *args, **kwargs):
                    if os.fspath(path) == str(leaf):
                        raise OSError(error_number, "injected leaf inspection failure", str(leaf))
                    return original_lstat(path, *args, **kwargs)

                def recorded_resolve(path, *args, **kwargs):
                    resolved.append(path)
                    return original_resolve(path, *args, **kwargs)

                with patch.object(os, "lstat", new=failing_lstat):
                    # Compare under the same filesystem failure, not merely
                    # against a result from before the failure was injected.
                    try:
                        expected = CONFIG._canonical_image({"image": "image.jpg"}, str(self.images))
                    except OSError as error:
                        with patch.object(Path, "resolve", new=recorded_resolve), self.assertRaises(type(error)):
                            CONFIG._ImagePathResolver()({"image": "image.jpg"}, str(self.images))
                    else:
                        with patch.object(Path, "resolve", new=recorded_resolve):
                            actual = CONFIG._ImagePathResolver()({"image": "image.jpg"}, str(self.images))
                        self.assertEqual(actual, expected)
                self.assertIn(leaf, resolved)

    def test_regular_siblings_resolve_parent_once_and_cache_complete_paths(self):
        records = [{"image": f"{index}.jpg"} for index in range(8)]
        for record in records:
            (self.images / record["image"]).touch()
        original_resolve, original_lstat = Path.resolve, os.lstat
        resolved, inspected = [], []

        def recorded_resolve(path, *args, **kwargs):
            resolved.append(path)
            return original_resolve(path, *args, **kwargs)

        def recorded_lstat(path, *args, **kwargs):
            inspected.append(os.fspath(path))
            return original_lstat(path, *args, **kwargs)

        resolver = CONFIG._ImagePathResolver()
        with patch.object(Path, "resolve", new=recorded_resolve), patch.object(os, "lstat", new=recorded_lstat):
            for _ in range(4):
                for record in records:
                    self.assertEqual(resolver(record, str(self.images)), str(self.images / record["image"]))
        self.assertEqual(resolved, [self.images])
        for record in records:
            self.assertEqual(inspected.count(str(self.images / record["image"])), 1)

    def test_cache_keys_keep_folders_distinct_but_absolute_images_ignore_folder(self):
        other = self.base / "other-images"
        other.mkdir()
        for folder in (self.images, other):
            (folder / "shared.jpg").touch()
        resolver = CONFIG._ImagePathResolver()
        record = {"image": "shared.jpg"}
        self.assertNotEqual(resolver(record, str(self.images)), resolver(record, str(other)))
        absolute = {"image": str(self.images / "shared.jpg")}
        with patch.object(Path, "resolve", side_effect=AssertionError("complete-path cache missed")), \
                patch.object(os, "lstat", side_effect=AssertionError("complete-path cache missed")):
            self.assertEqual(resolver(absolute, str(other)), str(self.images / "shared.jpg"))

    def test_invalid_images_keep_validation_errors(self):
        for image in (None, "", 0, ["image.jpg"]):
            with self.subTest(image=image):
                for resolve in (CONFIG._canonical_image, CONFIG._ImagePathResolver()):
                    with self.assertRaisesRegex(ValueError, "nonempty single image path"):
                        resolve({"image": image}, str(self.images))


class SplitResolutionIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.images = self.base / "images"
        self.images.mkdir()
        self.path = self.base / "train.json"
        self.records = [{"id": index, "image": f"{index}.jpg",
                         "conversations": [{"value": f"caption {index}"}]} for index in range(12)]
        for record in self.records:
            (self.images / record["image"]).touch()
        self.path.write_text(json.dumps(self.records), encoding="utf-8")
        self.data = SimpleNamespace(data_path=str(self.path), image_folder=str(self.images),
                                    train_data_fraction=1.0, protected_cka_validation_data_path=None,
                                    protected_cka_validation_image_folder=None)
        self.training = args(output_dir=str(self.base / "optimized"), protected_cka_validation_fraction=0.25,
                             protected_cka_validation_max_samples=4)

    def prepare(self, training=None):
        dataset = FakeDataset(str(self.path), None, self.data)
        return CONFIG.prepare_protected_cka_datasets(dataset, None, self.data,
                                                    training or self.training, FakeDataset)

    def test_optional_helper_resolver_matches_legacy_defaults(self):
        (self.images / "alias.jpg").symlink_to("0.jpg")
        records = self.records + [{"id": 99, "image": "alias.jpg"}]
        resolver = CONFIG._ImagePathResolver()
        folder = str(self.images)
        self.assertEqual(CONFIG._identity(records[0], folder, resolver), CONFIG._identity(records[0], folder))
        self.assertEqual(CONFIG._fingerprint(records, folder, str(self.path), resolver),
                         CONFIG._fingerprint(records, folder, str(self.path)))
        self.assertEqual(CONFIG._groups(records, folder, resolver), CONFIG._groups(records, folder))
        self.assertEqual(CONFIG._manifest(records, range(len(records)), folder, resolver),
                         CONFIG._manifest(records, range(len(records)), folder))
        CONFIG._assert_disjoint(records[:1], records[1:2], folder, folder, resolver)
        with self.assertRaisesRegex(ValueError, "overlap"):
            CONFIG._assert_disjoint(records[:1], records[-1:], folder, folder, resolver)

    def test_full_split_manifests_and_fingerprint_are_byte_identical_to_legacy(self):
        (self.images / "alias.jpg").symlink_to("0.jpg")
        self.records[1]["image"] = "alias.jpg"
        self.records[2]["id"] = 1  # Preserve transitive duplicate grouping too.
        self.path.write_text(json.dumps(self.records), encoding="utf-8")
        optimized_train, optimized_validation = self.prepare()
        baseline = args(**dict(vars(self.training), output_dir=str(self.base / "legacy")))
        with patch.object(CONFIG, "_ImagePathResolver", return_value=CONFIG._canonical_image):
            legacy_train, legacy_validation = self.prepare(baseline)
        self.assertEqual(optimized_train.list_data_dict, legacy_train.list_data_dict)
        self.assertEqual(optimized_validation.list_data_dict, legacy_validation.list_data_dict)
        self.assertEqual(self.training.protected_cka_split_fingerprint, baseline.protected_cka_split_fingerprint)
        for filename in ("split.json", "train_manifest.json", "validation_manifest.json"):
            self.assertEqual((Path(self.training.output_dir) / "protected_cka_split" / filename).read_bytes(),
                             (Path(baseline.output_dir) / "protected_cka_split" / filename).read_bytes())

    def test_all_split_phases_share_one_invocation_cache(self):
        original_resolve, original_lstat = Path.resolve, os.lstat
        resolved, inspected = [], []

        def recorded_resolve(path, *args, **kwargs):
            resolved.append(path)
            return original_resolve(path, *args, **kwargs)

        def recorded_lstat(path, *args, **kwargs):
            inspected.append(os.fspath(path))
            return original_lstat(path, *args, **kwargs)

        with patch.object(Path, "resolve", new=recorded_resolve), patch.object(os, "lstat", new=recorded_lstat):
            self.prepare()
        self.assertEqual(resolved.count(self.images), 1)
        for record in self.records:
            self.assertEqual(inspected.count(str(self.images / record["image"])), 1)

    def test_fresh_cache_detects_leaf_symlink_mutation_between_invocations(self):
        alias = self.images / "alias.jpg"
        alias.symlink_to("0.jpg")
        self.records[1]["image"] = "alias.jpg"
        self.path.write_text(json.dumps(self.records), encoding="utf-8")
        self.prepare()
        alias.unlink()
        alias.symlink_to("2.jpg")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.prepare()

    def test_fresh_cache_detects_parent_symlink_mutation_between_invocations(self):
        other = self.base / "other-images"
        other.mkdir()
        for record in self.records:
            (other / record["image"]).touch()
        alias = self.base / "image-root"
        alias.symlink_to(self.images, target_is_directory=True)
        self.data.image_folder = str(alias)
        self.prepare()
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.prepare()

    def test_explicit_validation_distinct_folders_and_symlink_overlap(self):
        validation_folder = self.base / "validation-images"
        validation_folder.mkdir()
        (validation_folder / "0.jpg").touch()
        validation_path = self.base / "validation.json"
        validation_path.write_text(json.dumps([{"id": 99, "image": "0.jpg"}]), encoding="utf-8")
        self.data.protected_cka_validation_data_path = str(validation_path)
        self.data.protected_cka_validation_image_folder = str(validation_folder)
        train, validation = self.prepare()
        self.assertEqual(len(train.list_data_dict), 12)
        self.assertEqual(validation.list_data_dict, [{"id": 99, "image": "0.jpg"}])
        # Identical basenames in different roots are distinct, but an actual
        # leaf symlink into the training root must still be treated as leakage.
        (validation_folder / "0.jpg").unlink()
        (validation_folder / "0.jpg").symlink_to(self.images / "0.jpg")
        self.training.output_dir = str(self.base / "overlapping")
        with self.assertRaisesRegex(ValueError, "overlap by image"):
            self.prepare()

    def test_disabled_is_noop_without_constructing_resolver_or_reading_paths(self):
        self.training.protected_cka_refinement = False
        dataset = FakeDataset(str(self.path), None, self.data)
        original_rows = dataset.list_data_dict
        with patch.object(CONFIG, "_ImagePathResolver", side_effect=AssertionError("resolver constructed")), \
                patch.object(Path, "resolve", side_effect=AssertionError("path resolution attempted")):
            train, validation = CONFIG.prepare_protected_cka_datasets(
                dataset, None, self.data, self.training, FakeDataset)
        self.assertIs(train, dataset)
        self.assertIs(train.list_data_dict, original_rows)
        self.assertIsNone(validation)
        self.assertFalse(Path(self.training.output_dir).exists())


if __name__ == "__main__":
    unittest.main()
