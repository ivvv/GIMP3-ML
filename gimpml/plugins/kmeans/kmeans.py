#!/usr/bin/env python3
# coding: utf-8
"""
 .d8888b.  8888888 888b     d888 8888888b.       888b     d888 888
d88P  Y88b   888   8888b   d8888 888   Y88b      8888b   d8888 888
888    888   888   88888b.d88888 888    888      88888b.d88888 888
888          888   888Y88888P888 888   d88P      888Y88888P888 888
888  88888   888   888 Y888P 888 8888888P"       888 Y888P 888 888
888    888   888   888  Y8P  888 888             888  Y8P  888 888
Y88b  d88P   888   888   "   888 888             888   "   888 888
 "Y8888P88 8888888 888       888 888             888       888 88888888

Performs K-means clustering for the current layer.
"""
import gi
gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp, GimpUi, GObject, GLib, Gio, Gtk
import gettext, subprocess, os, sys, json
sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")])
from plugin_utils import *
from module_utils import *
from constants import *

_ = gettext.gettext


def k_means(
    procedure, image, drawables, n_cluster, position, progress_bar, config_path_output
):
    # Save inference parameters and layers
    weight_path = config_path_output["weight_path"]
    python_path = config_path_output["python_path"]
    plugin_path = config_path_output["plugin_path"]

    undo_group_start(image)
    init_tmp()

    save_image(BASE_IMG, image, drawables[0])
    set_model_config({
            "n_cluster": int(n_cluster),
            "position": bool(position),
            "inference_status": "started",
        }, PLUGIN_ID)

    # Run inference and load as layer
    subprocess.call([python_path, plugin_path])
    data_output = get_model_config(PLUGIN_ID)

    if data_output["inference_status"] == "success":
        load_single_result_and_insert(RESULT_IMG, image, "K-means")
        undo_group_end(image)
        cleanup_tmp() # Remove temporary images
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
    else:
        undo_group_end(image)
        elog = "See GIMP console for error text"
        if "last_error" in data_output: elog = data_output["last_error"]
        show_dialog(
            "Interpolation was not performed due to errors.\nError text:\n" + elog,
            "Error!", "error", image_paths
        )
        cleanup_tmp() # Remove temporary images
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


def run(procedure, run_mode, image, n_drawables, layer, args, data):
    n_cluster = args.index(0)
    position = args.index(1)

    if run_mode == Gimp.RunMode.INTERACTIVE:
        # Get all paths
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools"
        )
        config_path_output = get_config()
        python_path = config_path_output["python_path"]
        config_path_output["plugin_path"] = os.path.join(config_path, "kmeans.py")

        config = procedure.create_config()
        config.begin_run(image, run_mode, args)

        GimpUi.init("kmeans.py")
        use_header_bar = Gtk.Settings.get_default().get_property(
            "gtk-dialogs-use-header"
        )
        dialog = GimpUi.Dialog(use_header_bar=use_header_bar, title=_("K Means..."))
        dialog.add_button(_("_Cancel"), Gtk.ResponseType.CANCEL)
        dialog.add_button(_("_Help"), Gtk.ResponseType.APPLY)
        dialog.add_button(_("_Process"), Gtk.ResponseType.OK)

        vbox = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, homogeneous=False, spacing=10
        )
        dialog.get_content_area().add(vbox)
        vbox.show()

        # Create grid to set all the properties inside.
        grid = Gtk.Grid()
        grid.set_column_homogeneous(False)
        grid.set_border_width(10)
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        vbox.add(grid)
        grid.show()

        # Show Logo
        logo = Gtk.Image.new_from_file(image_paths["logo"])
        # grid.attach(logo, 0, 0, 1, 1)
        vbox.pack_start(logo, False, False, 1)
        logo.show()

        # Show License
        license_text = _(PLUGIN_LICENSE)
        label = Gtk.Label(label=license_text)
        # grid.attach(label, 1, 1, 1, 1)
        vbox.pack_start(label, False, False, 1)
        label.show()

        # n_cluster parameter
        label = Gtk.Label.new_with_mnemonic(_("_Clusters"))
        grid.attach(label, 0, 0, 1, 1)
        label.show()
        spin = GimpUi.prop_spin_button_new(
            config, "n_cluster", step_increment=1, page_increment=10, digits=0
        )
        grid.attach(spin, 1, 0, 1, 1)
        spin.show()

        # Sample average parameter
        spin = GimpUi.prop_check_button_new(config, "position", _("Use _Position"))
        spin.set_tooltip_text(
            _(
                "If checked, x, y coordinates will be used as features for k means clustering"
            )
        )
        grid.attach(spin, 2, 0, 1, 1)
        spin.show()

        progress_bar = Gtk.ProgressBar()
        vbox.add(progress_bar)
        progress_bar.show()

        # Wait for user to click
        dialog.show()
        while True:
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                n_cluster = config.get_property("n_cluster")
                position = config.get_property("position")
                result = k_means(
                    procedure,
                    image,
                    layer,
                    n_cluster,
                    position,
                    progress_bar,
                    config_path_output,
                )
                # If the execution was successful, save parameters so they will be restored next time we show dialog.
                if result.index(0) == Gimp.PDBStatusType.SUCCESS and config is not None:
                    config.end_run(Gimp.PDBStatusType.SUCCESS)
                return result
            elif response == Gtk.ResponseType.APPLY:
                url = HELP_URL + "item-7-11"
                Gio.app_info_launch_default_for_uri(url, None)
                continue
            else:
                dialog.destroy()
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )


class Kmeans(Gimp.PlugIn):
    ## Parameters ##
    __gproperties__ = {
        "n_cluster": (
            float,
            _("_Clusters"),
            "Number of clusters",
            3,
            64,
            5,
            GObject.ParamFlags.READWRITE,
        ),
        "position": (
            bool,
            _("Use _Position"),
            "Use as position of pixels",
            True,
            GObject.ParamFlags.READWRITE,
        ),
    }

    ## GimpPlugIn virtual methods ##
    def do_query_procedures(self):
        return [PLUGIN_ID]
        
    def do_set_i18n(self, procname):
        return False, 'gimp30-python', None

    def do_create_procedure(self, name):
        procedure = None
        if name == PLUGIN_ID:
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, run, None
            )

            procedure.set_image_types("*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_documentation(
                N_("Performs K-means clustering for the current layer."),
                globals()["__doc__"],  # Include docstring from filestart
                name,
            )
            procedure.set_menu_label(N_("_K Means..."))
            procedure.set_attribution(*ATTRIBUTION_INFO)
            procedure.add_menu_path(MENU_LOCATION)
            procedure.add_argument_from_property(self, "n_cluster")
            procedure.add_argument_from_property(self, "position")

        return procedure


Gimp.main(Kmeans.__gtype__, sys.argv)
