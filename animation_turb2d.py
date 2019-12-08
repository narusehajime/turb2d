#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'CSV Reader'
grdcsv = CSVReader(FileName=['/home/naruse/test_images_turb2d/test_largeflow/grd.csv.00', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.01', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.02', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.03', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.04', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.05', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.06', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.07', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.08', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.09', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.10', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.11', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.12', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.13', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.14', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.15', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.16', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.17', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.18', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.19', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.20', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.21', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.22', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.23', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.24', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.25', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.26', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.27', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.28', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.29', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.30', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.31', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.32', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.33', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.34', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.35', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.36', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.37', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.38', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.39', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.40', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.41', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.42', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.43', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.44', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.45', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.46', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.47', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.48', '/home/naruse/test_images_turb2d/test_largeflow/grd.csv.49'])

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Create a new 'SpreadSheet View'
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024
# uncomment following to set a specific view size
# spreadSheetView1.ViewSize = [400, 400]

# get layout
layout1 = GetLayout()

# place view in the layout
layout1.AssignView(2, spreadSheetView1)

# show data in view
grdcsvDisplay = Show(grdcsv, spreadSheetView1)
# trace defaults for the display properties.
grdcsvDisplay.FieldAssociation = 'Row Data'

# find view
renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [610, 940]

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
spreadSheetView1.Update()

# create a new 'Table To Structured Grid'
tableToStructuredGrid1 = TableToStructuredGrid(Input=grdcsv)
tableToStructuredGrid1.XColumn = ''
tableToStructuredGrid1.YColumn = ''
tableToStructuredGrid1.ZColumn = ''

# Properties modified on tableToStructuredGrid1
tableToStructuredGrid1.WholeExtent = [0, 99, 0, 399, 0, 0]
tableToStructuredGrid1.XColumn = 'X'
tableToStructuredGrid1.YColumn = 'Y'
tableToStructuredGrid1.ZColumn = 'Z'

# show data in view
tableToStructuredGrid1Display = Show(tableToStructuredGrid1, spreadSheetView1)

# hide data in view
Hide(grdcsv, spreadSheetView1)

# update the view to ensure updated data information
spreadSheetView1.Update()

# show data in view
tableToStructuredGrid1Display_1 = Show(tableToStructuredGrid1, renderView1)
# trace defaults for the display properties.
tableToStructuredGrid1Display_1.Representation = 'Surface'
tableToStructuredGrid1Display_1.ColorArrayName = [None, '']
tableToStructuredGrid1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
tableToStructuredGrid1Display_1.ScaleFactor = 399.0
tableToStructuredGrid1Display_1.GlyphType = 'Arrow'
tableToStructuredGrid1Display_1.DataAxesGrid = 'GridAxesRepresentation'
tableToStructuredGrid1Display_1.PolarAxes = 'PolarAxesRepresentation'
tableToStructuredGrid1Display_1.ScalarOpacityUnitDistance = 121.0288461933739

# reset view to fit data
renderView1.ResetCamera()

# change solid color
tableToStructuredGrid1Display_1.DiffuseColor = [0.7215686274509804, 0.7215686274509804, 0.7215686274509804]

# reset view to fit data
renderView1.ResetCamera()

# set active source
SetActiveSource(grdcsv)

# create a new 'Table To Structured Grid'
tableToStructuredGrid2 = TableToStructuredGrid(Input=grdcsv)
tableToStructuredGrid2.XColumn = ''
tableToStructuredGrid2.YColumn = ''
tableToStructuredGrid2.ZColumn = ''

# Properties modified on tableToStructuredGrid2
tableToStructuredGrid2.WholeExtent = [0, 99, 0, 399, 0, 0]
tableToStructuredGrid2.XColumn = 'X'
tableToStructuredGrid2.YColumn = 'Y'
tableToStructuredGrid2.ZColumn = 'xi'

# show data in view
tableToStructuredGrid2Display = Show(tableToStructuredGrid2, renderView1)
# trace defaults for the display properties.
tableToStructuredGrid2Display.Representation = 'Surface'
tableToStructuredGrid2Display.ColorArrayName = [None, '']
tableToStructuredGrid2Display.OSPRayScaleFunction = 'PiecewiseFunction'
tableToStructuredGrid2Display.ScaleFactor = 399.0
tableToStructuredGrid2Display.GlyphType = 'Arrow'
tableToStructuredGrid2Display.DataAxesGrid = 'GridAxesRepresentation'
tableToStructuredGrid2Display.PolarAxes = 'PolarAxesRepresentation'
tableToStructuredGrid2Display.ScalarOpacityUnitDistance = 121.0288461933739

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(tableToStructuredGrid2Display, ('POINTS', 'C'))

# rescale color and/or opacity maps used to include current data range
tableToStructuredGrid2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
tableToStructuredGrid2Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'C'
cLUT = GetColorTransferFunction('C')
cLUT.EnableOpacityMapping = 1
cLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 0.0036, 0.865003, 0.865003, 0.865003, 0.0072, 0.705882, 0.0156863, 0.14902]
cLUT.ScalarRangeInitialized = 1.0

# set active view
SetActiveView(spreadSheetView1)

# destroy spreadSheetView1
Delete(spreadSheetView1)
del spreadSheetView1

# close an empty frame
layout1.Collapse(2)

# set active view
SetActiveView(renderView1)

# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(Input=tableToStructuredGrid2)

# show data in view
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 17

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 16

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 15

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 14

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 13

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 12

# current camera placement for renderView1
renderView1.CameraPosition = [-2768.2699860329494, -3026.1382099684097, 3351.660459933763]
renderView1.CameraFocalPoint = [495.0, 1995.0, 149.5]
renderView1.CameraViewUp = [0.201207965551656, 0.400421355111962, 0.893967612930616]
renderView1.CameraParallelScale = 3175.84730049397

# save animation
SaveAnimation('/home/naruse/test_images_turb2d/test_largeflow/test.avi', renderView1, ImageResolution=[1228, 940],
    FrameWindow=[0, 49])

animationScene1.Play()

# current camera placement for renderView1
renderView1.CameraPosition = [-2768.2699860329494, -3026.1382099684097, 3351.660459933763]
renderView1.CameraFocalPoint = [495.0, 1995.0, 149.5]
renderView1.CameraViewUp = [0.201207965551656, 0.400421355111962, 0.893967612930616]
renderView1.CameraParallelScale = 3175.84730049397

# save animation
SaveAnimation('/home/naruse/test_images_turb2d/test_largeflow/test_largeflow.png', renderView1, ImageResolution=[1229, 940],
    FrameWindow=[0, 49])

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [-2768.2699860329494, -3026.1382099684097, 3351.660459933763]
renderView1.CameraFocalPoint = [495.0, 1995.0, 149.5]
renderView1.CameraViewUp = [0.201207965551656, 0.400421355111962, 0.893967612930616]
renderView1.CameraParallelScale = 3175.84730049397

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
