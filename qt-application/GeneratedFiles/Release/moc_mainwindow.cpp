/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created: Mon 27. May 12:20:04 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../mainwindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_mainwindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      34,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      12,   11,   11,   11, 0x08,
      26,   11,   11,   11, 0x08,
      40,   11,   11,   11, 0x08,
      58,   11,   11,   11, 0x08,
      81,   11,   11,   11, 0x08,
      98,   11,   11,   11, 0x08,
     116,   11,   11,   11, 0x08,
     136,   11,   11,   11, 0x08,
     166,   11,   11,   11, 0x08,
     193,   11,   11,   11, 0x08,
     207,   11,   11,   11, 0x08,
     224,   11,   11,   11, 0x08,
     245,   11,   11,   11, 0x08,
     271,   11,   11,   11, 0x08,
     295,   11,   11,   11, 0x08,
     310,   11,   11,   11, 0x08,
     337,   11,   11,   11, 0x08,
     356,   11,   11,   11, 0x08,
     383,   11,   11,   11, 0x08,
     403,   11,   11,   11, 0x08,
     420,   11,   11,   11, 0x08,
     435,   11,   11,   11, 0x08,
     451,   11,   11,   11, 0x08,
     467,   11,   11,   11, 0x08,
     482,   11,   11,   11, 0x08,
     501,   11,   11,   11, 0x08,
     519,   11,   11,   11, 0x08,
     544,   11,   11,   11, 0x08,
     571,   11,   11,   11, 0x08,
     586,   11,   11,   11, 0x08,
     609,   11,   11,   11, 0x08,
     634,   11,   11,   11, 0x08,
     662,  656,   11,   11, 0x08,
     681,   11,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_mainwindow[] = {
    "mainwindow\0\0onAboutHytm()\0updateStuff()\0"
    "showTerrainInfo()\0onSimulationSettings()\0"
    "onRainSettings()\0onWaterSettings()\0"
    "onErosionSettings()\0onInteractiveMarkerSettings()\0"
    "onConstantMarkerSettings()\0onAddSource()\0"
    "onDeleteSource()\0onDeleteAllSources()\0"
    "onCurrentSourceSettings()\0"
    "onSourceListSelection()\0onGLSettings()\0"
    "onGLSelectWaterBaseColor()\0"
    "onGLSelectMarker()\0onGLSelectConstantMarker()\0"
    "onGLSelectBGColor()\0onTerrainScale()\0"
    "onTerrainNew()\0onTerrainLoad()\0"
    "onTerrainSave()\0onTerrainAdd()\0"
    "onTerrainSaveAll()\0onTerrainDelete()\0"
    "onTerrainMaterialColor()\0"
    "onTerrainMaterialDensity()\0onResetWater()\0"
    "onLayerListSelection()\0onCurrentLayerSettings()\0"
    "onCurrentLayerColor()\0delta\0"
    "onBrushScroll(int)\0onSourceButton()\0"
};

void mainwindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        mainwindow *_t = static_cast<mainwindow *>(_o);
        switch (_id) {
        case 0: _t->onAboutHytm(); break;
        case 1: _t->updateStuff(); break;
        case 2: _t->showTerrainInfo(); break;
        case 3: _t->onSimulationSettings(); break;
        case 4: _t->onRainSettings(); break;
        case 5: _t->onWaterSettings(); break;
        case 6: _t->onErosionSettings(); break;
        case 7: _t->onInteractiveMarkerSettings(); break;
        case 8: _t->onConstantMarkerSettings(); break;
        case 9: _t->onAddSource(); break;
        case 10: _t->onDeleteSource(); break;
        case 11: _t->onDeleteAllSources(); break;
        case 12: _t->onCurrentSourceSettings(); break;
        case 13: _t->onSourceListSelection(); break;
        case 14: _t->onGLSettings(); break;
        case 15: _t->onGLSelectWaterBaseColor(); break;
        case 16: _t->onGLSelectMarker(); break;
        case 17: _t->onGLSelectConstantMarker(); break;
        case 18: _t->onGLSelectBGColor(); break;
        case 19: _t->onTerrainScale(); break;
        case 20: _t->onTerrainNew(); break;
        case 21: _t->onTerrainLoad(); break;
        case 22: _t->onTerrainSave(); break;
        case 23: _t->onTerrainAdd(); break;
        case 24: _t->onTerrainSaveAll(); break;
        case 25: _t->onTerrainDelete(); break;
        case 26: _t->onTerrainMaterialColor(); break;
        case 27: _t->onTerrainMaterialDensity(); break;
        case 28: _t->onResetWater(); break;
        case 29: _t->onLayerListSelection(); break;
        case 30: _t->onCurrentLayerSettings(); break;
        case 31: _t->onCurrentLayerColor(); break;
        case 32: _t->onBrushScroll((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 33: _t->onSourceButton(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData mainwindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject mainwindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_mainwindow,
      qt_meta_data_mainwindow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &mainwindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *mainwindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *mainwindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_mainwindow))
        return static_cast<void*>(const_cast< mainwindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int mainwindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 34)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 34;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
