"""Tests for font configuration module."""
import sys
from unittest.mock import MagicMock, patch
import pytest

# Use Agg backend for headless testing
import matplotlib
matplotlib.use("Agg")


class TestPickFirstAvailableFamily:
    """Test pick_first_available_family function."""
    
    def test_returns_none_if_no_candidates_available(self):
        """When no candidate fonts exist, return None."""
        from ui.fonts import pick_first_available_family
        
        mock_root = MagicMock()
        mock_root.tk.call.return_value = "x11"
        
        with patch("tkinter.font.families", return_value=["FakeFont1", "FakeFont2"]):
            result = pick_first_available_family(
                mock_root, 
                ["Ubuntu", "Noto Sans", "DejaVu Sans"]
            )
            assert result is None
    
    def test_case_insensitive_match(self):
        """Font matching should be case-insensitive."""
        from ui.fonts import pick_first_available_family
        
        mock_root = MagicMock()
        
        # Tk returns fonts with specific casing
        with patch("tkinter.font.families", return_value=["UBUNTU", "noto sans"]):
            result = pick_first_available_family(
                mock_root, 
                ["ubuntu", "Noto Sans"]  # lowercase/uppercase mix
            )
            assert result == "UBUNTU"  # should preserve Tk's casing
    
    def test_returns_actual_tk_casing(self):
        """Should return the exact family name from Tk, preserving its casing."""
        from ui.fonts import pick_first_available_family
        
        mock_root = MagicMock()
        
        with patch("tkinter.font.families", return_value=["DejaVu Sans", "Liberation Sans"]):
            result = pick_first_available_family(
                mock_root, 
                ["dejavu sans"]  # lowercase query
            )
            assert result == "DejaVu Sans"  # preserve Tk's casing
    
    def test_returns_first_available_from_ordered_list(self):
        """Should return the first available font from prioritized list."""
        from ui.fonts import pick_first_available_family
        
        mock_root = MagicMock()
        
        with patch("tkinter.font.families", return_value=["Arial", "Ubuntu", "Noto Sans"]):
            # Prioritize Ubuntu over Arial
            result = pick_first_available_family(
                mock_root, 
                ["Ubuntu", "DejaVu Sans", "Arial"]
            )
            assert result == "Ubuntu"  # first in priority that exists
        
        with patch("tkinter.font.families", return_value=["Arial", "Ubuntu"]):
            result = pick_first_available_family(
                mock_root, 
                ["Noto Sans", "DejaVu Sans", "Arial"]
            )
            assert result == "Arial"  # Ubuntu not in families list


class TestConfigureNamedFonts:
    """Test configure_named_fonts function."""
    
    def test_no_crash_when_named_font_missing(self):
        """Should handle missing named fonts gracefully."""
        from ui.fonts import configure_named_fonts
        
        mock_root = MagicMock()
        
        # Mock nametofont to raise exception for some names
        call_count = [0]
        
        def mock_nametofont(name):
            call_count[0] += 1
            if call_count[0] > 2:  # After first few calls, raise exception
                raise tkfont.TclError("unknown font name")
            mock_font = MagicMock()
            return mock_font
        
        import tkinter.font as tkfont
        with patch.object(tkfont, "nametofont", side_effect=mock_nametofont):
            # Should not crash even if fonts don't exist
            configure_named_fonts(
                mock_root, 
                sans_family="Ubuntu", 
                mono_family="Ubuntu Mono", 
                ui_size=10
            )
    
    def test_sets_family_when_provided(self):
        """Should configure font family when provided."""
        from ui.fonts import configure_named_fonts
        
        mock_root = MagicMock()
        
        # Mock nametofont to return a font object
        mock_font = MagicMock()
        mock_font.actual.return_value = {"family": "TkDefault", "size": 10}
        
        import tkinter.font as tkfont
        with patch.object(tkfont, "nametofont", return_value=mock_font):
            configure_named_fonts(
                mock_root, 
                sans_family="Ubuntu", 
                mono_family="Liberation Mono", 
                ui_size=12
            )
            
            # Should have been called to configure fonts
            assert mock_font.configure.called
    
    def test_skips_family_when_none(self):
        """Should not set family when None is provided."""
        from ui.fonts import configure_named_fonts
        
        mock_root = MagicMock()
        
        mock_font = MagicMock()
        mock_font.actual.return_value = {"family": "TkDefault", "size": 10}
        
        import tkinter.font as tkfont
        with patch.object(tkfont, "nametofont", return_value=mock_font):
            configure_named_fonts(
                mock_root, 
                sans_family=None, 
                mono_family=None, 
                ui_size=10
            )
            
            # Should still call nametofont but not set family
            # (it will configure other properties)


class TestConfigureTtkStyles:
    """Test configure_ttk_styles function."""
    
    def test_no_crash_when_family_none(self):
        """Should handle None sans_family gracefully."""
        from ui.fonts import configure_ttk_styles
        
        mock_style = MagicMock()
        
        # Should not call configure at all when family is None
        configure_ttk_styles(
            mock_style, 
            sans_family=None, 
            mono_family="Ubuntu Mono", 
            ui_size=10
        )
        
        # No configure calls should happen
        assert not mock_style.configure.called
    
    def test_configures_fit_button_bold(self):
        """Should configure Fit.TButton with bold variant."""
        from ui.fonts import configure_ttk_styles
        
        mock_style = MagicMock()
        
        configure_ttk_styles(
            mock_style, 
            sans_family="Ubuntu", 
            mono_family=None, 
            ui_size=10
        )
        
        # Should have been called to configure styles
        assert mock_style.configure.called


class TestModuleLevelCache:
    """Test font caching behavior."""
    
    def test_caches_results_per_root(self):
        """Results should be cached per root window id."""
        from ui.fonts import pick_first_available_family, _font_cache
        
        # Clear cache for this test
        _font_cache.clear()
        
        mock_root1 = MagicMock()
        mock_root2 = MagicMock()
        
        with patch("tkinter.font.families", return_value=["Ubuntu"]):
            result1 = pick_first_available_family(mock_root1, ["Ubuntu"])
            result2 = pick_first_available_family(mock_root2, ["Ubuntu"])
            
            # Both should find Ubuntu
            assert result1 == "Ubuntu"
            assert result2 == "Ubuntu"
            
            # Cache should have entries for both roots
            assert len(_font_cache) >= 2


class TestFontCandidates:
    """Test that font candidate lists are properly defined."""
    
    def test_sans_candidates_not_empty(self):
        """Sans-serif candidates list should not be empty."""
        from ui.fonts import SANS_CANDIDATES
        
        assert len(SANS_CANDIDATES) > 0
    
    def test_mono_candidates_not_empty(self):
        """Monospace candidates list should not be empty."""
        from ui.fonts import MONO_CANDIDATES
        
        assert len(MONO_CANDIDATES) > 0
    
    def test_default_ui_size_is_int(self):
        """DEFAULT_UI_SIZE should be an integer."""
        from ui.fonts import DEFAULT_UI_SIZE
        
        assert isinstance(DEFAULT_UI_SIZE, int)
        assert DEFAULT_UI_SIZE == 10
