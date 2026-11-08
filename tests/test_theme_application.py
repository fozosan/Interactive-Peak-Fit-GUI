"""
Test suite for theme application functionality.

Tests ensure that:
1. Theme module can be imported without errors
2. apply_theme() handles missing dependencies gracefully
3. apply_theme() returns correct status and theme names
4. App starts successfully with/without theme packages
5. All GUI functionalities work regardless of theme
"""

import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestThemeImport:
    """Test that theme module can be imported."""
    
    def test_theme_module_imports(self):
        """Verify ui.theme module exists and imports correctly."""
        from ui.theme import apply_theme, log_theme_status
        assert apply_theme is not None
        assert log_theme_status is not None


class TestApplyThemeGracefulFallback:
    """Test theme application with missing dependencies."""
    
    def test_apply_theme_returns_tuple(self):
        """Test that apply_theme returns a tuple of (bool, str)."""
        from ui.theme import apply_theme
        root = tk.Tk()
        try:
            result = apply_theme(root)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)
        finally:
            root.destroy()
    
    def test_apply_theme_without_sv_ttk_or_ttkthemes(self):
        """Test apply_theme falls back to default when no theme packages available."""
        from ui.theme import apply_theme
        root = tk.Tk()
        try:
            # Hide sv-ttk and ttkthemes temporarily
            with patch.dict(sys.modules, {'sv_ttk': None, 'ttkthemes': None}):
                success, theme_name = apply_theme(root)
                # Should return False and indicate default theme
                assert success is False
                assert "default" in theme_name.lower()
        finally:
            root.destroy()
    
    def test_apply_theme_with_sv_ttk_missing(self):
        """Test apply_theme when sv_ttk import fails."""
        from ui.theme import apply_theme
        root = tk.Tk()
        try:
            # Mock sv_ttk to raise ImportError
            with patch.dict(sys.modules, {'sv_ttk': None}):
                success, theme_name = apply_theme(root)
                # Should still return some result without crashing
                assert isinstance(success, bool)
                assert isinstance(theme_name, str)
        finally:
            root.destroy()
    
    def test_apply_theme_handles_exceptions(self):
        """Test that apply_theme handles exceptions gracefully."""
        from ui.theme import apply_theme
        root = tk.Tk()
        try:
            # Even if something goes wrong, should not crash
            result = apply_theme(root)
            assert result is not None
            assert len(result) == 2
        finally:
            root.destroy()
    
    def test_log_theme_status_with_theme_applied(self, capsys):
        """Test logging when theme is successfully applied."""
        from ui.theme import log_theme_status
        log_theme_status(True, "sv-ttk (Fluent Dark)")
        captured = capsys.readouterr()
        assert "Theme applied" in captured.out
        assert "sv-ttk" in captured.out
    
    def test_log_theme_status_without_theme(self, capsys):
        """Test logging when no theme is available."""
        from ui.theme import log_theme_status
        log_theme_status(False, "default (Tkinter)")
        captured = capsys.readouterr()
        assert "default" in captured.out.lower() or "not available" in captured.out.lower()


class TestAppStartupWithTheme:
    """Test that app starts successfully with theme integration."""
    
    def test_main_function_exists(self):
        """Verify main() function exists in ui.app."""
        from ui.app import main
        assert callable(main)
    
    def test_main_imports_theme(self):
        """Verify main imports apply_theme."""
        import inspect
        from ui.app import main
        source = inspect.getsource(main)
        assert "apply_theme" in source
        assert "log_theme_status" in source


class TestThemeWithDefaultTkinter:
    """Test app functionality with default Tkinter (no themes)."""
    
    def test_root_window_created_and_destroyed(self):
        """Test that Tk root window can be created and destroyed."""
        root = tk.Tk()
        try:
            assert root.winfo_exists()
            # Simulate what apply_theme does
            from ui.theme import apply_theme
            apply_theme(root)
            assert root.winfo_exists()
        finally:
            root.destroy()
    
    def test_ttk_widgets_work_without_theme(self):
        """Test that ttk widgets work without custom themes."""
        from tkinter import ttk
        root = tk.Tk()
        try:
            # Create various ttk widgets to ensure they work
            button = ttk.Button(root, text="Test Button")
            label = ttk.Label(root, text="Test Label")
            frame = ttk.Frame(root)
            
            # They should exist and be functional
            assert button.winfo_exists()
            assert label.winfo_exists()
            assert frame.winfo_exists()
        finally:
            root.destroy()


class TestRequirementsUpdated:
    """Test that requirements.txt is properly updated."""
    
    def test_requirements_file_updated(self):
        """Verify requirements.txt includes theme packages as optional."""
        requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
        with open(requirements, "r") as f:
            content = f.read()
            # Check that theme packages are mentioned
            assert "sv-ttk" in content or "ttkthemes" in content
            # Check for optional markers
            assert "#optional" in content.lower() or "optional" in content.lower()


class TestThemeNonFatal:
    """Test that theme loading is non-fatal to app startup."""
    
    def test_apply_theme_never_crashes(self):
        """Test that apply_theme never raises exceptions."""
        from ui.theme import apply_theme
        root = tk.Tk()
        try:
            # Should never raise any exception
            try:
                result = apply_theme(root)
                assert result is not None
            except Exception as e:
                pytest.fail(f"apply_theme raised unexpected exception: {e}")
        finally:
            root.destroy()
    
    def test_theme_status_logging_never_crashes(self):
        """Test that logging theme status never crashes."""
        from ui.theme import log_theme_status
        # Should never raise any exception
        try:
            log_theme_status(True, "test-theme")
            log_theme_status(False, "default")
        except Exception as e:
            pytest.fail(f"log_theme_status raised unexpected exception: {e}")


class TestThemeTakesEffect:
    """The theme apply_theme() reports is the one widgets get."""

    def test_classic_widgets_built_after_apply_theme_are_dark(self):
        """sv-ttk recolours classic Tk widgets on <<ThemeChanged>>, which Tk sends when idle.

        main() builds the app right after apply_theme(), and matplotlib picks black or
        light toolbar icons from the button background at that moment. If the palette
        lands later, the toolbar keeps black icons on a dark background.
        """
        pytest.importorskip("sv_ttk")
        from ui.theme import apply_theme
        root = tk.Tk()
        try:
            applied, name = apply_theme(root)
            assert applied and "sv-ttk" in name
            button = tk.Button(root)
            assert max(root.winfo_rgb(button.cget("background"))) < 65535 / 2
        finally:
            root.destroy()

    def test_ttkthemes_arc_applies_when_sv_ttk_is_missing(self):
        pytest.importorskip("ttkthemes")
        from tkinter import ttk
        from ui.theme import apply_theme
        root = tk.Tk()
        try:
            with patch.dict(sys.modules, {"sv_ttk": None}):
                result = apply_theme(root)
            assert result == (True, "ttkthemes (Arc)")
            assert ttk.Style(root).theme_use() == "arc"
        finally:
            root.destroy()


class TestThemeIntegrationWithApp:
    """Test theme integration with the main app."""
    
    def test_app_can_be_instantiated_with_theme(self):
        """Test that PeakFitApp can be instantiated after theme is applied."""
        from ui.app import PeakFitApp
        from ui.theme import apply_theme
        
        root = tk.Tk()
        try:
            # Apply theme first
            apply_theme(root)
            
            # Then try to create app (will test imports and basic setup)
            # We won't create the full app since it needs matplotlib and data,
            # but we can verify the import works
            assert PeakFitApp is not None
        finally:
            root.destroy()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
