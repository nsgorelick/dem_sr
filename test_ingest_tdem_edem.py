import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from ingest_tdem_edem import (
    DONE_STAGE,
    DOWNLOAD_PENDING_STAGE,
    EXTRACT_PENDING_STAGE,
    FAILED_STAGE,
    GcsTarget,
    SharedInventory,
    StateLedger,
    TileItem,
    apply_work_limit,
    build_tile_items,
    classify_tile,
    cleanup_local_artifacts,
    extract_primary_tif_from_remote_zip,
    parse_earthengine_ls_output,
    parse_gsutil_ls_output,
    select_primary_dem_member,
    tile_id_from_url,
    tile_paths,
)


class IngestTdemEdemTests(unittest.TestCase):
    def test_tile_id_from_url(self) -> None:
        url = (
            "https://download.geoservice.dlr.de/TDM30_EDEM/files/"
            "foo/TDM1_EDEM_10_N50E013_V01_C.zip"
        )
        self.assertEqual(tile_id_from_url(url), "TDM1_EDEM_10_N50E013_V01_C")

    def test_build_tile_items_uses_tile_stem_for_names(self) -> None:
        url = (
            "https://download.geoservice.dlr.de/TDM30_EDEM/files/"
            "foo/TDM1_EDEM_10_N50E013_V01_C.zip"
        )
        item = build_tile_items(
            [url],
            GcsTarget(bucket="ee-gorelick-upload", prefix="tdem_edem/tifs"),
            "users/ngorelick/DTM/TDEM_EDEM",
            Path("workspace"),
        )[0]
        self.assertEqual(
            item.gcs_uri,
            "gs://ee-gorelick-upload/tdem_edem/tifs/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM.tif",
        )
        self.assertEqual(
            item.asset_id,
            "users/ngorelick/DTM/TDEM_EDEM/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM",
        )

    def test_select_primary_dem_member_prefers_main_dem(self) -> None:
        tile_id = "TDM1_EDEM_10_N50E013_V01_C"
        names = [
            "nested/TDM1_EDEM_10_N50E013_V01_C_HEM.tif",
            "nested/TDM1_EDEM_10_N50E013_EDEM_W84.tif",
            "nested/TDM1_EDEM_10_N50E013_EDEM_EGM.tif",
            "nested/TDM1_EDEM_10_N50E013_V01_C_EDM.tif",
        ]
        self.assertEqual(
            select_primary_dem_member(tile_id, names),
            "nested/TDM1_EDEM_10_N50E013_EDEM_EGM.tif",
        )

    def test_select_primary_dem_member_requires_egm(self) -> None:
        tile_id = "TDM1_EDEM_10_N50E013_V01_C"
        names = [
            "nested/TDM1_EDEM_10_N50E013_EDEM_W84.tif",
            "nested/TDM1_EDEM_10_N50E013_EDM.tif",
        ]
        with self.assertRaises(FileNotFoundError):
            select_primary_dem_member(tile_id, names)

    def test_parse_gsutil_ls_output(self) -> None:
        stdout = "\n".join(
            [
                "gs://ee-gorelick-upload/tdem_edem/tifs/TDM1_EDEM_10_N50E013_V01_C.tif",
                "gs://ee-gorelick-upload/tdem_edem/tifs/TDM1_EDEM_10_N50E014_V01_C_EDEM_EGM.tif",
            ]
        )
        self.assertEqual(
            parse_gsutil_ls_output(stdout),
            {"TDM1_EDEM_10_N50E013_V01_C", "TDM1_EDEM_10_N50E014_V01_C"},
        )

    def test_parse_earthengine_ls_output(self) -> None:
        stdout = "\n".join(
            [
                "users/ngorelick/DTM/TDEM_EDEM/TDM1_EDEM_10_N50E013_V01_C",
                "users/ngorelick/DTM/TDEM_EDEM/TDM1_EDEM_10_N50E014_V01_C_EDEM_EGM",
            ]
        )
        self.assertEqual(
            parse_earthengine_ls_output(stdout),
            {"TDM1_EDEM_10_N50E013_V01_C", "TDM1_EDEM_10_N50E014_V01_C"},
        )

    def test_classify_tile_prefers_remote_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tile_id = "TDM1_EDEM_10_N50E013_V01_C"
            item = TileItem(
                tile_id=tile_id,
                output_stem=f"{tile_id}_EDEM_EGM",
                url="https://example.com/TDM1_EDEM_10_N50E013_V01_C.zip",
                gcs_uri="gs://bucket/prefix/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM.tif",
                gcs_object_name="prefix/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM.tif",
                asset_id="users/ngorelick/DTM/TDEM_EDEM/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM",
                paths=tile_paths(workspace, tile_id, f"{tile_id}_EDEM_EGM"),
            )
            self.assertEqual(
                classify_tile(item, {}, SharedInventory(set(), {tile_id})),
                DONE_STAGE,
            )
            self.assertEqual(
                classify_tile(item, {}, SharedInventory({tile_id}, set())),
                "ingest_pending",
            )

    def test_classify_tile_uses_local_stage_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tile_id = "TDM1_EDEM_10_N50E013_V01_C"
            item = TileItem(
                tile_id=tile_id,
                output_stem=f"{tile_id}_EDEM_EGM",
                url="https://example.com/TDM1_EDEM_10_N50E013_V01_C.zip",
                gcs_uri="gs://bucket/prefix/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM.tif",
                gcs_object_name="prefix/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM.tif",
                asset_id="users/ngorelick/DTM/TDEM_EDEM/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM",
                paths=tile_paths(workspace, tile_id, f"{tile_id}_EDEM_EGM"),
            )
            self.assertEqual(
                classify_tile(item, {}, SharedInventory(set(), set())),
                DOWNLOAD_PENDING_STAGE,
            )

            item.paths.zip_path.parent.mkdir(parents=True, exist_ok=True)
            item.paths.zip_path.write_bytes(b"zip")
            self.assertEqual(
                classify_tile(item, {}, SharedInventory(set(), set())),
                EXTRACT_PENDING_STAGE,
            )

            item.paths.raw_tif_path.parent.mkdir(parents=True, exist_ok=True)
            item.paths.raw_tif_path.write_bytes(b"raw")
            self.assertEqual(
                classify_tile(item, {}, SharedInventory(set(), set())),
                "normalize_pending",
            )

            item.paths.fixed_tif_path.parent.mkdir(parents=True, exist_ok=True)
            item.paths.fixed_tif_path.write_bytes(b"fixed")
            self.assertEqual(
                classify_tile(item, {}, SharedInventory(set(), set())),
                "upload_pending",
            )

    def test_classify_tile_preserves_failed_ledger_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tile_id = "TDM1_EDEM_10_N50E013_V01_C"
            item = TileItem(
                tile_id=tile_id,
                output_stem=f"{tile_id}_EDEM_EGM",
                url="https://example.com/TDM1_EDEM_10_N50E013_V01_C.zip",
                gcs_uri="gs://bucket/prefix/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM.tif",
                gcs_object_name="prefix/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM.tif",
                asset_id="users/ngorelick/DTM/TDEM_EDEM/TDM1_EDEM_10_N50E013_V01_C_EDEM_EGM",
                paths=tile_paths(workspace, tile_id, f"{tile_id}_EDEM_EGM"),
            )
            self.assertEqual(
                classify_tile(item, {"stage": FAILED_STAGE}, SharedInventory(set(), set())),
                FAILED_STAGE,
            )

    def test_state_ledger_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = StateLedger(Path(tmpdir) / "state.json")
            ledger.update("tile_a", stage="uploading", operation_name="op-123")
            reloaded = StateLedger(Path(tmpdir) / "state.json")
            self.assertEqual(reloaded.get("tile_a")["stage"], "uploading")
            self.assertEqual(reloaded.get("tile_a")["operation_name"], "op-123")

    def test_apply_work_limit_counts_only_pending_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            items = [
                TileItem(
                    tile_id=f"tile_{idx}",
                    output_stem=f"tile_{idx}_EDEM_EGM",
                    url=f"https://example.com/tile_{idx}.zip",
                    gcs_uri=f"gs://bucket/prefix/tile_{idx}_EDEM_EGM.tif",
                    gcs_object_name=f"prefix/tile_{idx}_EDEM_EGM.tif",
                    asset_id=f"users/ngorelick/DTM/TDEM_EDEM/tile_{idx}_EDEM_EGM",
                    paths=tile_paths(workspace, f"tile_{idx}", f"tile_{idx}_EDEM_EGM"),
                )
                for idx in range(4)
            ]
            stage_by_tile = {
                "tile_0": DONE_STAGE,
                "tile_1": "ingest_pending",
                "tile_2": "upload_pending",
                "tile_3": DONE_STAGE,
            }
            limited = apply_work_limit(items, stage_by_tile, 1)
            self.assertEqual([item.tile_id for item in limited], ["tile_1"])

    def test_cleanup_local_artifacts_removes_files_and_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = tile_paths(Path(tmpdir), "tile_a")
            for path in (
                paths.zip_path,
                paths.raw_tif_path,
                paths.fixed_tif_path,
                paths.manifest_path,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
                path.with_suffix(path.suffix + ".part").write_text("x", encoding="utf-8")
            cleanup_local_artifacts(paths)
            for path in (
                paths.zip_path,
                paths.raw_tif_path,
                paths.fixed_tif_path,
                paths.manifest_path,
            ):
                self.assertFalse(path.exists())
                self.assertFalse(path.with_suffix(path.suffix + ".part").exists())

    def test_extract_primary_tif_from_remote_zip_writes_only_egm_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "raw" / "tile.tif"
            auth = Mock()
            auth.request_headers.return_value = {"Authorization": "Basic test"}

            class FakeMember:
                def __enter__(self):
                    from io import BytesIO

                    return BytesIO(b"egm-bytes")

                def __exit__(self, exc_type, exc, tb):
                    return False

            class FakeRemoteZip:
                def __init__(self, url, headers, timeout, initial_buffer_size):
                    self.url = url
                    self.headers = headers
                    self.timeout = timeout
                    self.initial_buffer_size = initial_buffer_size

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def namelist(self):
                    return [
                        "folder/TDM1_EDEM_10_N50E013_EDEM_W84.tif",
                        "folder/TDM1_EDEM_10_N50E013_EDEM_EGM.tif",
                    ]

                def open(self, member):
                    assert member.endswith("EDEM_EGM.tif")
                    return FakeMember()

            member = extract_primary_tif_from_remote_zip(
                "https://example.com/file.zip",
                "TDM1_EDEM_10_N50E013_V01_C",
                output_path,
                auth=auth,
                timeout=10.0,
                remotezip_cls=FakeRemoteZip,
            )
            self.assertEqual(member, "folder/TDM1_EDEM_10_N50E013_EDEM_EGM.tif")
            self.assertEqual(output_path.read_bytes(), b"egm-bytes")


if __name__ == "__main__":
    unittest.main()
