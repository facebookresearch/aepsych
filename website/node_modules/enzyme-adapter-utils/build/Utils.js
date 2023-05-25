"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
Object.defineProperty(exports, "RootFinder", {
  enumerable: true,
  get: function get() {
    return _RootFinder["default"];
  }
});
exports.assertDomAvailable = assertDomAvailable;
exports.compareNodeTypeOf = compareNodeTypeOf;
Object.defineProperty(exports, "createMountWrapper", {
  enumerable: true,
  get: function get() {
    return _createMountWrapper["default"];
  }
});
Object.defineProperty(exports, "createRenderWrapper", {
  enumerable: true,
  get: function get() {
    return _createRenderWrapper["default"];
  }
});
exports.displayNameOfNode = displayNameOfNode;
exports.elementToTree = elementToTree;
exports.ensureKeyOrUndefined = ensureKeyOrUndefined;
exports.fakeDynamicImport = fakeDynamicImport;
exports.findElement = findElement;
exports.flatten = flatten;
exports.getComponentStack = getComponentStack;
exports.getMaskedContext = getMaskedContext;
exports.getNodeFromRootFinder = getNodeFromRootFinder;
exports.getWrappingComponentMountRenderer = getWrappingComponentMountRenderer;
exports.isArrayLike = isArrayLike;
exports.mapNativeEventNames = mapNativeEventNames;
exports.nodeTypeFromType = nodeTypeFromType;
exports.propFromEvent = propFromEvent;
exports.propsWithKeysAndRef = propsWithKeysAndRef;
exports.simulateError = simulateError;
exports.spyMethod = spyMethod;
exports.spyProperty = spyProperty;
exports.withSetStateAllowed = withSetStateAllowed;
Object.defineProperty(exports, "wrap", {
  enumerable: true,
  get: function get() {
    return _wrapWithSimpleWrapper["default"];
  }
});
exports.wrapWithWrappingComponent = wrapWithWrappingComponent;
var _functionPrototype = _interopRequireDefault(require("function.prototype.name"));
var _object = _interopRequireDefault(require("object.fromentries"));
var _has = _interopRequireDefault(require("has"));
var _createMountWrapper = _interopRequireDefault(require("./createMountWrapper"));
var _createRenderWrapper = _interopRequireDefault(require("./createRenderWrapper"));
var _wrapWithSimpleWrapper = _interopRequireDefault(require("./wrapWithSimpleWrapper"));
var _RootFinder = _interopRequireDefault(require("./RootFinder"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }
function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }
function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }
function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }
function _iterableToArrayLimit(arr, i) { var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"]; if (_i == null) return; var _arr = []; var _n = true; var _d = false; var _s, _e; try { for (_i = _i.call(arr); !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }
function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }
function _typeof(obj) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, _typeof(obj); }
function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ownKeys(Object(source), !0).forEach(function (key) { _defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function mapNativeEventNames(event) {
  var _ref = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {},
    _ref$animation = _ref.animation,
    animation = _ref$animation === void 0 ? false : _ref$animation,
    _ref$pointerEvents = _ref.pointerEvents,
    pointerEvents = _ref$pointerEvents === void 0 ? false : _ref$pointerEvents,
    _ref$auxClick = _ref.auxClick,
    auxClick = _ref$auxClick === void 0 ? false : _ref$auxClick;
  var nativeToReactEventMap = _objectSpread(_objectSpread(_objectSpread({
    compositionend: 'compositionEnd',
    compositionstart: 'compositionStart',
    compositionupdate: 'compositionUpdate',
    keydown: 'keyDown',
    keyup: 'keyUp',
    keypress: 'keyPress',
    contextmenu: 'contextMenu',
    dblclick: 'doubleClick',
    doubleclick: 'doubleClick',
    // kept for legacy. TODO: remove with next major.
    dragend: 'dragEnd',
    dragenter: 'dragEnter',
    dragexist: 'dragExit',
    dragleave: 'dragLeave',
    dragover: 'dragOver',
    dragstart: 'dragStart',
    mousedown: 'mouseDown',
    mousemove: 'mouseMove',
    mouseout: 'mouseOut',
    mouseover: 'mouseOver',
    mouseup: 'mouseUp',
    touchcancel: 'touchCancel',
    touchend: 'touchEnd',
    touchmove: 'touchMove',
    touchstart: 'touchStart',
    canplay: 'canPlay',
    canplaythrough: 'canPlayThrough',
    durationchange: 'durationChange',
    loadeddata: 'loadedData',
    loadedmetadata: 'loadedMetadata',
    loadstart: 'loadStart',
    ratechange: 'rateChange',
    timeupdate: 'timeUpdate',
    volumechange: 'volumeChange',
    beforeinput: 'beforeInput',
    mouseenter: 'mouseEnter',
    mouseleave: 'mouseLeave',
    transitionend: 'transitionEnd'
  }, animation && {
    animationstart: 'animationStart',
    animationiteration: 'animationIteration',
    animationend: 'animationEnd'
  }), pointerEvents && {
    pointerdown: 'pointerDown',
    pointermove: 'pointerMove',
    pointerup: 'pointerUp',
    pointercancel: 'pointerCancel',
    gotpointercapture: 'gotPointerCapture',
    lostpointercapture: 'lostPointerCapture',
    pointerenter: 'pointerEnter',
    pointerleave: 'pointerLeave',
    pointerover: 'pointerOver',
    pointerout: 'pointerOut'
  }), auxClick && {
    auxclick: 'auxClick'
  });
  return nativeToReactEventMap[event] || event;
}

// 'click' => 'onClick'
// 'mouseEnter' => 'onMouseEnter'
function propFromEvent(event) {
  var eventOptions = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
  var nativeEvent = mapNativeEventNames(event, eventOptions);
  return "on".concat(nativeEvent[0].toUpperCase()).concat(nativeEvent.slice(1));
}
function withSetStateAllowed(fn) {
  // NOTE(lmr):
  // this is currently here to circumvent a React bug where `setState()` is
  // not allowed without global being defined.
  var cleanup = false;
  if (typeof global.document === 'undefined') {
    cleanup = true;
    global.document = {};
  }
  var result = fn();
  if (cleanup) {
    // This works around a bug in node/jest in that developers aren't able to
    // delete things from global when running in a node vm.
    global.document = undefined;
    delete global.document;
  }
  return result;
}
function assertDomAvailable(feature) {
  if (!global || !global.document || !global.document.createElement) {
    throw new Error("Enzyme's ".concat(feature, " expects a DOM environment to be loaded, but found none"));
  }
}
function displayNameOfNode(node) {
  if (!node) return null;
  var type = node.type;
  if (!type) return null;
  return type.displayName || (typeof type === 'function' ? (0, _functionPrototype["default"])(type) : type.name || type);
}
function nodeTypeFromType(type) {
  if (typeof type === 'string') {
    return 'host';
  }
  if (type && type.prototype && type.prototype.isReactComponent) {
    return 'class';
  }
  return 'function';
}
function getIteratorFn(obj) {
  var iteratorFn = obj && (typeof Symbol === 'function' && _typeof(Symbol.iterator) === 'symbol' && obj[Symbol.iterator] || obj['@@iterator']);
  if (typeof iteratorFn === 'function') {
    return iteratorFn;
  }
  return undefined;
}
function isIterable(obj) {
  return !!getIteratorFn(obj);
}
function isArrayLike(obj) {
  return Array.isArray(obj) || typeof obj !== 'string' && isIterable(obj);
}
function flatten(arrs) {
  // optimize for the most common case
  if (Array.isArray(arrs)) {
    return arrs.reduce(function (flatArrs, item) {
      return flatArrs.concat(isArrayLike(item) ? flatten(item) : item);
    }, []);
  }

  // fallback for arbitrary iterable children
  var flatArrs = [];
  var iteratorFn = getIteratorFn(arrs);
  var iterator = iteratorFn.call(arrs);
  var step = iterator.next();
  while (!step.done) {
    var item = step.value;
    var flatItem = void 0;
    if (isArrayLike(item)) {
      flatItem = flatten(item);
    } else {
      flatItem = item;
    }
    flatArrs = flatArrs.concat(flatItem);
    step = iterator.next();
  }
  return flatArrs;
}
function ensureKeyOrUndefined(key) {
  return key || (key === '' ? '' : undefined);
}
function elementToTree(el) {
  var recurse = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : elementToTree;
  if (typeof recurse !== 'function' && arguments.length === 3) {
    // special case for backwards compat for `.map(elementToTree)`
    recurse = elementToTree; // eslint-disable-line no-param-reassign
  }

  if (el === null || _typeof(el) !== 'object' || !('type' in el)) {
    return el;
  }
  var type = el.type,
    props = el.props,
    key = el.key,
    ref = el.ref;
  var children = props.children;
  var rendered = null;
  if (isArrayLike(children)) {
    rendered = flatten(children).map(function (x) {
      return recurse(x);
    });
  } else if (typeof children !== 'undefined') {
    rendered = recurse(children);
  }
  var nodeType = nodeTypeFromType(type);
  if (nodeType === 'host' && props.dangerouslySetInnerHTML) {
    if (props.children != null) {
      var error = new Error('Can only set one of `children` or `props.dangerouslySetInnerHTML`.');
      error.name = 'Invariant Violation';
      throw error;
    }
  }
  return {
    nodeType: nodeType,
    type: type,
    props: props,
    key: ensureKeyOrUndefined(key),
    ref: ref,
    instance: null,
    rendered: rendered
  };
}
function mapFind(arraylike, mapper, finder) {
  var found;
  var isFound = Array.prototype.find.call(arraylike, function (item) {
    found = mapper(item);
    return finder(found);
  });
  return isFound ? found : undefined;
}
function findElement(el, predicate) {
  if (el === null || _typeof(el) !== 'object' || !('type' in el)) {
    return undefined;
  }
  if (predicate(el)) {
    return el;
  }
  var rendered = el.rendered;
  if (isArrayLike(rendered)) {
    return mapFind(rendered, function (x) {
      return findElement(x, predicate);
    }, function (x) {
      return typeof x !== 'undefined';
    });
  }
  return findElement(rendered, predicate);
}
function propsWithKeysAndRef(node) {
  if (node.ref !== null || node.key !== null) {
    return _objectSpread(_objectSpread({}, node.props), {}, {
      key: node.key,
      ref: node.ref
    });
  }
  return node.props;
}
function getComponentStack(hierarchy) {
  var getNodeType = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : nodeTypeFromType;
  var getDisplayName = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : displayNameOfNode;
  var tuples = hierarchy.filter(function (node) {
    return node.type !== _RootFinder["default"];
  }).map(function (x) {
    return [getNodeType(x.type), getDisplayName(x)];
  }).concat([['class', 'WrapperComponent']]);
  return tuples.map(function (_ref2, i, arr) {
    var _ref3 = _slicedToArray(_ref2, 2),
      name = _ref3[1];
    var _ref4 = arr.slice(i + 1).find(function (_ref6) {
        var _ref7 = _slicedToArray(_ref6, 1),
          nodeType = _ref7[0];
        return nodeType !== 'host';
      }) || [],
      _ref5 = _slicedToArray(_ref4, 2),
      closestComponent = _ref5[1];
    return "\n    in ".concat(name).concat(closestComponent ? " (created by ".concat(closestComponent, ")") : '');
  }).join('');
}
function simulateError(error, catchingInstance, rootNode,
// TODO: remove `rootNode` next semver-major
hierarchy) {
  var getNodeType = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : nodeTypeFromType;
  var getDisplayName = arguments.length > 5 && arguments[5] !== undefined ? arguments[5] : displayNameOfNode;
  var catchingType = arguments.length > 6 && arguments[6] !== undefined ? arguments[6] : {};
  var instance = catchingInstance || {};
  var componentDidCatch = instance.componentDidCatch;
  var getDerivedStateFromError = catchingType.getDerivedStateFromError;
  if (!componentDidCatch && !getDerivedStateFromError) {
    throw error;
  }
  if (getDerivedStateFromError) {
    var stateUpdate = getDerivedStateFromError.call(catchingType, error);
    instance.setState(stateUpdate);
  }
  if (componentDidCatch) {
    var componentStack = getComponentStack(hierarchy, getNodeType, getDisplayName);
    componentDidCatch.call(instance, error, {
      componentStack: componentStack
    });
  }
}
function getMaskedContext(contextTypes, unmaskedContext) {
  if (!contextTypes || !unmaskedContext) {
    return {};
  }
  return (0, _object["default"])(Object.keys(contextTypes).map(function (key) {
    return [key, unmaskedContext[key]];
  }));
}
function getNodeFromRootFinder(isCustomComponent, tree, options) {
  if (!isCustomComponent(options.wrappingComponent)) {
    return tree.rendered;
  }
  var rootFinder = findElement(tree, function (node) {
    return node.type === _RootFinder["default"];
  });
  if (!rootFinder) {
    throw new Error('`wrappingComponent` must render its children!');
  }
  return rootFinder.rendered;
}
function wrapWithWrappingComponent(createElement, node, options) {
  var wrappingComponent = options.wrappingComponent,
    wrappingComponentProps = options.wrappingComponentProps;
  if (!wrappingComponent) {
    return node;
  }
  return createElement(wrappingComponent, wrappingComponentProps, createElement(_RootFinder["default"], null, node));
}
function getWrappingComponentMountRenderer(_ref8) {
  var toTree = _ref8.toTree,
    getMountWrapperInstance = _ref8.getMountWrapperInstance;
  return {
    getNode: function getNode() {
      var instance = getMountWrapperInstance();
      return instance ? toTree(instance).rendered : null;
    },
    render: function render(el, context, callback) {
      var instance = getMountWrapperInstance();
      if (!instance) {
        throw new Error('The wrapping component may not be updated if the root is unmounted.');
      }
      return instance.setWrappingComponentProps(el.props, callback);
    }
  };
}
function fakeDynamicImport(moduleToImport) {
  return Promise.resolve({
    "default": moduleToImport
  });
}
function compareNodeTypeOf(node, matchingTypeOf) {
  if (!node) {
    return false;
  }
  return node.$$typeof === matchingTypeOf;
}

// TODO: when enzyme v3.12.0 is required, delete this
function spyMethod(instance, methodName) {
  var getStub = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : function () {};
  var lastReturnValue;
  var originalMethod = instance[methodName];
  var hasOwn = (0, _has["default"])(instance, methodName);
  var descriptor;
  if (hasOwn) {
    descriptor = Object.getOwnPropertyDescriptor(instance, methodName);
  }
  Object.defineProperty(instance, methodName, {
    configurable: true,
    enumerable: !descriptor || !!descriptor.enumerable,
    value: getStub(originalMethod) || function spied() {
      for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
        args[_key] = arguments[_key];
      }
      var result = originalMethod.apply(this, args);
      lastReturnValue = result;
      return result;
    }
  });
  return {
    restore: function restore() {
      if (hasOwn) {
        if (descriptor) {
          Object.defineProperty(instance, methodName, descriptor);
        } else {
          /* eslint-disable no-param-reassign */
          instance[methodName] = originalMethod;
          /* eslint-enable no-param-reassign */
        }
      } else {
        /* eslint-disable no-param-reassign */
        delete instance[methodName];
        /* eslint-enable no-param-reassign */
      }
    },
    getLastReturnValue: function getLastReturnValue() {
      return lastReturnValue;
    }
  };
}

// TODO: when enzyme v3.12.0 is required, delete this
function spyProperty(instance, propertyName) {
  var handlers = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
  var originalValue = instance[propertyName];
  var hasOwn = (0, _has["default"])(instance, propertyName);
  var descriptor;
  if (hasOwn) {
    descriptor = Object.getOwnPropertyDescriptor(instance, propertyName);
  }
  var _wasAssigned = false;
  var holder = originalValue;
  var getV = handlers.get ? function () {
    var value = descriptor && descriptor.get ? descriptor.get.call(instance) : holder;
    return handlers.get.call(instance, value);
  } : function () {
    return holder;
  };
  var set = handlers.set ? function (newValue) {
    _wasAssigned = true;
    var handlerNewValue = handlers.set.call(instance, holder, newValue);
    holder = handlerNewValue;
    if (descriptor && descriptor.set) {
      descriptor.set.call(instance, holder);
    }
  } : function (v) {
    _wasAssigned = true;
    holder = v;
  };
  Object.defineProperty(instance, propertyName, {
    configurable: true,
    enumerable: !descriptor || !!descriptor.enumerable,
    get: getV,
    set: set
  });
  return {
    restore: function restore() {
      if (hasOwn) {
        if (descriptor) {
          Object.defineProperty(instance, propertyName, descriptor);
        } else {
          /* eslint-disable no-param-reassign */
          instance[propertyName] = holder;
          /* eslint-enable no-param-reassign */
        }
      } else {
        /* eslint-disable no-param-reassign */
        delete instance[propertyName];
        /* eslint-enable no-param-reassign */
      }
    },
    wasAssigned: function wasAssigned() {
      return _wasAssigned;
    }
  };
}
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJuYW1lcyI6WyJtYXBOYXRpdmVFdmVudE5hbWVzIiwiZXZlbnQiLCJhbmltYXRpb24iLCJwb2ludGVyRXZlbnRzIiwiYXV4Q2xpY2siLCJuYXRpdmVUb1JlYWN0RXZlbnRNYXAiLCJjb21wb3NpdGlvbmVuZCIsImNvbXBvc2l0aW9uc3RhcnQiLCJjb21wb3NpdGlvbnVwZGF0ZSIsImtleWRvd24iLCJrZXl1cCIsImtleXByZXNzIiwiY29udGV4dG1lbnUiLCJkYmxjbGljayIsImRvdWJsZWNsaWNrIiwiZHJhZ2VuZCIsImRyYWdlbnRlciIsImRyYWdleGlzdCIsImRyYWdsZWF2ZSIsImRyYWdvdmVyIiwiZHJhZ3N0YXJ0IiwibW91c2Vkb3duIiwibW91c2Vtb3ZlIiwibW91c2VvdXQiLCJtb3VzZW92ZXIiLCJtb3VzZXVwIiwidG91Y2hjYW5jZWwiLCJ0b3VjaGVuZCIsInRvdWNobW92ZSIsInRvdWNoc3RhcnQiLCJjYW5wbGF5IiwiY2FucGxheXRocm91Z2giLCJkdXJhdGlvbmNoYW5nZSIsImxvYWRlZGRhdGEiLCJsb2FkZWRtZXRhZGF0YSIsImxvYWRzdGFydCIsInJhdGVjaGFuZ2UiLCJ0aW1ldXBkYXRlIiwidm9sdW1lY2hhbmdlIiwiYmVmb3JlaW5wdXQiLCJtb3VzZWVudGVyIiwibW91c2VsZWF2ZSIsInRyYW5zaXRpb25lbmQiLCJhbmltYXRpb25zdGFydCIsImFuaW1hdGlvbml0ZXJhdGlvbiIsImFuaW1hdGlvbmVuZCIsInBvaW50ZXJkb3duIiwicG9pbnRlcm1vdmUiLCJwb2ludGVydXAiLCJwb2ludGVyY2FuY2VsIiwiZ290cG9pbnRlcmNhcHR1cmUiLCJsb3N0cG9pbnRlcmNhcHR1cmUiLCJwb2ludGVyZW50ZXIiLCJwb2ludGVybGVhdmUiLCJwb2ludGVyb3ZlciIsInBvaW50ZXJvdXQiLCJhdXhjbGljayIsInByb3BGcm9tRXZlbnQiLCJldmVudE9wdGlvbnMiLCJuYXRpdmVFdmVudCIsInRvVXBwZXJDYXNlIiwic2xpY2UiLCJ3aXRoU2V0U3RhdGVBbGxvd2VkIiwiZm4iLCJjbGVhbnVwIiwiZ2xvYmFsIiwiZG9jdW1lbnQiLCJyZXN1bHQiLCJ1bmRlZmluZWQiLCJhc3NlcnREb21BdmFpbGFibGUiLCJmZWF0dXJlIiwiY3JlYXRlRWxlbWVudCIsIkVycm9yIiwiZGlzcGxheU5hbWVPZk5vZGUiLCJub2RlIiwidHlwZSIsImRpc3BsYXlOYW1lIiwiZnVuY3Rpb25OYW1lIiwibmFtZSIsIm5vZGVUeXBlRnJvbVR5cGUiLCJwcm90b3R5cGUiLCJpc1JlYWN0Q29tcG9uZW50IiwiZ2V0SXRlcmF0b3JGbiIsIm9iaiIsIml0ZXJhdG9yRm4iLCJTeW1ib2wiLCJpdGVyYXRvciIsImlzSXRlcmFibGUiLCJpc0FycmF5TGlrZSIsIkFycmF5IiwiaXNBcnJheSIsImZsYXR0ZW4iLCJhcnJzIiwicmVkdWNlIiwiZmxhdEFycnMiLCJpdGVtIiwiY29uY2F0IiwiY2FsbCIsInN0ZXAiLCJuZXh0IiwiZG9uZSIsInZhbHVlIiwiZmxhdEl0ZW0iLCJlbnN1cmVLZXlPclVuZGVmaW5lZCIsImtleSIsImVsZW1lbnRUb1RyZWUiLCJlbCIsInJlY3Vyc2UiLCJhcmd1bWVudHMiLCJsZW5ndGgiLCJwcm9wcyIsInJlZiIsImNoaWxkcmVuIiwicmVuZGVyZWQiLCJtYXAiLCJ4Iiwibm9kZVR5cGUiLCJkYW5nZXJvdXNseVNldElubmVySFRNTCIsImVycm9yIiwiaW5zdGFuY2UiLCJtYXBGaW5kIiwiYXJyYXlsaWtlIiwibWFwcGVyIiwiZmluZGVyIiwiZm91bmQiLCJpc0ZvdW5kIiwiZmluZCIsImZpbmRFbGVtZW50IiwicHJlZGljYXRlIiwicHJvcHNXaXRoS2V5c0FuZFJlZiIsImdldENvbXBvbmVudFN0YWNrIiwiaGllcmFyY2h5IiwiZ2V0Tm9kZVR5cGUiLCJnZXREaXNwbGF5TmFtZSIsInR1cGxlcyIsImZpbHRlciIsIlJvb3RGaW5kZXIiLCJpIiwiYXJyIiwiY2xvc2VzdENvbXBvbmVudCIsImpvaW4iLCJzaW11bGF0ZUVycm9yIiwiY2F0Y2hpbmdJbnN0YW5jZSIsInJvb3ROb2RlIiwiY2F0Y2hpbmdUeXBlIiwiY29tcG9uZW50RGlkQ2F0Y2giLCJnZXREZXJpdmVkU3RhdGVGcm9tRXJyb3IiLCJzdGF0ZVVwZGF0ZSIsInNldFN0YXRlIiwiY29tcG9uZW50U3RhY2siLCJnZXRNYXNrZWRDb250ZXh0IiwiY29udGV4dFR5cGVzIiwidW5tYXNrZWRDb250ZXh0IiwiZnJvbUVudHJpZXMiLCJPYmplY3QiLCJrZXlzIiwiZ2V0Tm9kZUZyb21Sb290RmluZGVyIiwiaXNDdXN0b21Db21wb25lbnQiLCJ0cmVlIiwib3B0aW9ucyIsIndyYXBwaW5nQ29tcG9uZW50Iiwicm9vdEZpbmRlciIsIndyYXBXaXRoV3JhcHBpbmdDb21wb25lbnQiLCJ3cmFwcGluZ0NvbXBvbmVudFByb3BzIiwiZ2V0V3JhcHBpbmdDb21wb25lbnRNb3VudFJlbmRlcmVyIiwidG9UcmVlIiwiZ2V0TW91bnRXcmFwcGVySW5zdGFuY2UiLCJnZXROb2RlIiwicmVuZGVyIiwiY29udGV4dCIsImNhbGxiYWNrIiwic2V0V3JhcHBpbmdDb21wb25lbnRQcm9wcyIsImZha2VEeW5hbWljSW1wb3J0IiwibW9kdWxlVG9JbXBvcnQiLCJQcm9taXNlIiwicmVzb2x2ZSIsImNvbXBhcmVOb2RlVHlwZU9mIiwibWF0Y2hpbmdUeXBlT2YiLCIkJHR5cGVvZiIsInNweU1ldGhvZCIsIm1ldGhvZE5hbWUiLCJnZXRTdHViIiwibGFzdFJldHVyblZhbHVlIiwib3JpZ2luYWxNZXRob2QiLCJoYXNPd24iLCJoYXMiLCJkZXNjcmlwdG9yIiwiZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yIiwiZGVmaW5lUHJvcGVydHkiLCJjb25maWd1cmFibGUiLCJlbnVtZXJhYmxlIiwic3BpZWQiLCJhcmdzIiwiYXBwbHkiLCJyZXN0b3JlIiwiZ2V0TGFzdFJldHVyblZhbHVlIiwic3B5UHJvcGVydHkiLCJwcm9wZXJ0eU5hbWUiLCJoYW5kbGVycyIsIm9yaWdpbmFsVmFsdWUiLCJ3YXNBc3NpZ25lZCIsImhvbGRlciIsImdldFYiLCJnZXQiLCJzZXQiLCJuZXdWYWx1ZSIsImhhbmRsZXJOZXdWYWx1ZSIsInYiXSwic291cmNlcyI6WyIuLi9zcmMvVXRpbHMuanMiXSwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IGZ1bmN0aW9uTmFtZSBmcm9tICdmdW5jdGlvbi5wcm90b3R5cGUubmFtZSc7XG5pbXBvcnQgZnJvbUVudHJpZXMgZnJvbSAnb2JqZWN0LmZyb21lbnRyaWVzJztcbmltcG9ydCBoYXMgZnJvbSAnaGFzJztcbmltcG9ydCBjcmVhdGVNb3VudFdyYXBwZXIgZnJvbSAnLi9jcmVhdGVNb3VudFdyYXBwZXInO1xuaW1wb3J0IGNyZWF0ZVJlbmRlcldyYXBwZXIgZnJvbSAnLi9jcmVhdGVSZW5kZXJXcmFwcGVyJztcbmltcG9ydCB3cmFwIGZyb20gJy4vd3JhcFdpdGhTaW1wbGVXcmFwcGVyJztcbmltcG9ydCBSb290RmluZGVyIGZyb20gJy4vUm9vdEZpbmRlcic7XG5cbmV4cG9ydCB7XG4gIGNyZWF0ZU1vdW50V3JhcHBlcixcbiAgY3JlYXRlUmVuZGVyV3JhcHBlcixcbiAgd3JhcCxcbiAgUm9vdEZpbmRlcixcbn07XG5cbmV4cG9ydCBmdW5jdGlvbiBtYXBOYXRpdmVFdmVudE5hbWVzKGV2ZW50LCB7XG4gIGFuaW1hdGlvbiA9IGZhbHNlLCAvLyBzaG91bGQgYmUgdHJ1ZSBmb3IgUmVhY3QgMTUrXG4gIHBvaW50ZXJFdmVudHMgPSBmYWxzZSwgLy8gc2hvdWxkIGJlIHRydWUgZm9yIFJlYWN0IDE2LjQrXG4gIGF1eENsaWNrID0gZmFsc2UsIC8vIHNob3VsZCBiZSB0cnVlIGZvciBSZWFjdCAxNi41K1xufSA9IHt9KSB7XG4gIGNvbnN0IG5hdGl2ZVRvUmVhY3RFdmVudE1hcCA9IHtcbiAgICBjb21wb3NpdGlvbmVuZDogJ2NvbXBvc2l0aW9uRW5kJyxcbiAgICBjb21wb3NpdGlvbnN0YXJ0OiAnY29tcG9zaXRpb25TdGFydCcsXG4gICAgY29tcG9zaXRpb251cGRhdGU6ICdjb21wb3NpdGlvblVwZGF0ZScsXG4gICAga2V5ZG93bjogJ2tleURvd24nLFxuICAgIGtleXVwOiAna2V5VXAnLFxuICAgIGtleXByZXNzOiAna2V5UHJlc3MnLFxuICAgIGNvbnRleHRtZW51OiAnY29udGV4dE1lbnUnLFxuICAgIGRibGNsaWNrOiAnZG91YmxlQ2xpY2snLFxuICAgIGRvdWJsZWNsaWNrOiAnZG91YmxlQ2xpY2snLCAvLyBrZXB0IGZvciBsZWdhY3kuIFRPRE86IHJlbW92ZSB3aXRoIG5leHQgbWFqb3IuXG4gICAgZHJhZ2VuZDogJ2RyYWdFbmQnLFxuICAgIGRyYWdlbnRlcjogJ2RyYWdFbnRlcicsXG4gICAgZHJhZ2V4aXN0OiAnZHJhZ0V4aXQnLFxuICAgIGRyYWdsZWF2ZTogJ2RyYWdMZWF2ZScsXG4gICAgZHJhZ292ZXI6ICdkcmFnT3ZlcicsXG4gICAgZHJhZ3N0YXJ0OiAnZHJhZ1N0YXJ0JyxcbiAgICBtb3VzZWRvd246ICdtb3VzZURvd24nLFxuICAgIG1vdXNlbW92ZTogJ21vdXNlTW92ZScsXG4gICAgbW91c2VvdXQ6ICdtb3VzZU91dCcsXG4gICAgbW91c2VvdmVyOiAnbW91c2VPdmVyJyxcbiAgICBtb3VzZXVwOiAnbW91c2VVcCcsXG4gICAgdG91Y2hjYW5jZWw6ICd0b3VjaENhbmNlbCcsXG4gICAgdG91Y2hlbmQ6ICd0b3VjaEVuZCcsXG4gICAgdG91Y2htb3ZlOiAndG91Y2hNb3ZlJyxcbiAgICB0b3VjaHN0YXJ0OiAndG91Y2hTdGFydCcsXG4gICAgY2FucGxheTogJ2NhblBsYXknLFxuICAgIGNhbnBsYXl0aHJvdWdoOiAnY2FuUGxheVRocm91Z2gnLFxuICAgIGR1cmF0aW9uY2hhbmdlOiAnZHVyYXRpb25DaGFuZ2UnLFxuICAgIGxvYWRlZGRhdGE6ICdsb2FkZWREYXRhJyxcbiAgICBsb2FkZWRtZXRhZGF0YTogJ2xvYWRlZE1ldGFkYXRhJyxcbiAgICBsb2Fkc3RhcnQ6ICdsb2FkU3RhcnQnLFxuICAgIHJhdGVjaGFuZ2U6ICdyYXRlQ2hhbmdlJyxcbiAgICB0aW1ldXBkYXRlOiAndGltZVVwZGF0ZScsXG4gICAgdm9sdW1lY2hhbmdlOiAndm9sdW1lQ2hhbmdlJyxcbiAgICBiZWZvcmVpbnB1dDogJ2JlZm9yZUlucHV0JyxcbiAgICBtb3VzZWVudGVyOiAnbW91c2VFbnRlcicsXG4gICAgbW91c2VsZWF2ZTogJ21vdXNlTGVhdmUnLFxuICAgIHRyYW5zaXRpb25lbmQ6ICd0cmFuc2l0aW9uRW5kJyxcbiAgICAuLi4oYW5pbWF0aW9uICYmIHtcbiAgICAgIGFuaW1hdGlvbnN0YXJ0OiAnYW5pbWF0aW9uU3RhcnQnLFxuICAgICAgYW5pbWF0aW9uaXRlcmF0aW9uOiAnYW5pbWF0aW9uSXRlcmF0aW9uJyxcbiAgICAgIGFuaW1hdGlvbmVuZDogJ2FuaW1hdGlvbkVuZCcsXG4gICAgfSksXG4gICAgLi4uKHBvaW50ZXJFdmVudHMgJiYge1xuICAgICAgcG9pbnRlcmRvd246ICdwb2ludGVyRG93bicsXG4gICAgICBwb2ludGVybW92ZTogJ3BvaW50ZXJNb3ZlJyxcbiAgICAgIHBvaW50ZXJ1cDogJ3BvaW50ZXJVcCcsXG4gICAgICBwb2ludGVyY2FuY2VsOiAncG9pbnRlckNhbmNlbCcsXG4gICAgICBnb3Rwb2ludGVyY2FwdHVyZTogJ2dvdFBvaW50ZXJDYXB0dXJlJyxcbiAgICAgIGxvc3Rwb2ludGVyY2FwdHVyZTogJ2xvc3RQb2ludGVyQ2FwdHVyZScsXG4gICAgICBwb2ludGVyZW50ZXI6ICdwb2ludGVyRW50ZXInLFxuICAgICAgcG9pbnRlcmxlYXZlOiAncG9pbnRlckxlYXZlJyxcbiAgICAgIHBvaW50ZXJvdmVyOiAncG9pbnRlck92ZXInLFxuICAgICAgcG9pbnRlcm91dDogJ3BvaW50ZXJPdXQnLFxuICAgIH0pLFxuICAgIC4uLihhdXhDbGljayAmJiB7XG4gICAgICBhdXhjbGljazogJ2F1eENsaWNrJyxcbiAgICB9KSxcbiAgfTtcblxuICByZXR1cm4gbmF0aXZlVG9SZWFjdEV2ZW50TWFwW2V2ZW50XSB8fCBldmVudDtcbn1cblxuLy8gJ2NsaWNrJyA9PiAnb25DbGljaydcbi8vICdtb3VzZUVudGVyJyA9PiAnb25Nb3VzZUVudGVyJ1xuZXhwb3J0IGZ1bmN0aW9uIHByb3BGcm9tRXZlbnQoZXZlbnQsIGV2ZW50T3B0aW9ucyA9IHt9KSB7XG4gIGNvbnN0IG5hdGl2ZUV2ZW50ID0gbWFwTmF0aXZlRXZlbnROYW1lcyhldmVudCwgZXZlbnRPcHRpb25zKTtcbiAgcmV0dXJuIGBvbiR7bmF0aXZlRXZlbnRbMF0udG9VcHBlckNhc2UoKX0ke25hdGl2ZUV2ZW50LnNsaWNlKDEpfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB3aXRoU2V0U3RhdGVBbGxvd2VkKGZuKSB7XG4gIC8vIE5PVEUobG1yKTpcbiAgLy8gdGhpcyBpcyBjdXJyZW50bHkgaGVyZSB0byBjaXJjdW12ZW50IGEgUmVhY3QgYnVnIHdoZXJlIGBzZXRTdGF0ZSgpYCBpc1xuICAvLyBub3QgYWxsb3dlZCB3aXRob3V0IGdsb2JhbCBiZWluZyBkZWZpbmVkLlxuICBsZXQgY2xlYW51cCA9IGZhbHNlO1xuICBpZiAodHlwZW9mIGdsb2JhbC5kb2N1bWVudCA9PT0gJ3VuZGVmaW5lZCcpIHtcbiAgICBjbGVhbnVwID0gdHJ1ZTtcbiAgICBnbG9iYWwuZG9jdW1lbnQgPSB7fTtcbiAgfVxuICBjb25zdCByZXN1bHQgPSBmbigpO1xuICBpZiAoY2xlYW51cCkge1xuICAgIC8vIFRoaXMgd29ya3MgYXJvdW5kIGEgYnVnIGluIG5vZGUvamVzdCBpbiB0aGF0IGRldmVsb3BlcnMgYXJlbid0IGFibGUgdG9cbiAgICAvLyBkZWxldGUgdGhpbmdzIGZyb20gZ2xvYmFsIHdoZW4gcnVubmluZyBpbiBhIG5vZGUgdm0uXG4gICAgZ2xvYmFsLmRvY3VtZW50ID0gdW5kZWZpbmVkO1xuICAgIGRlbGV0ZSBnbG9iYWwuZG9jdW1lbnQ7XG4gIH1cbiAgcmV0dXJuIHJlc3VsdDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGFzc2VydERvbUF2YWlsYWJsZShmZWF0dXJlKSB7XG4gIGlmICghZ2xvYmFsIHx8ICFnbG9iYWwuZG9jdW1lbnQgfHwgIWdsb2JhbC5kb2N1bWVudC5jcmVhdGVFbGVtZW50KSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBFbnp5bWUncyAke2ZlYXR1cmV9IGV4cGVjdHMgYSBET00gZW52aXJvbm1lbnQgdG8gYmUgbG9hZGVkLCBidXQgZm91bmQgbm9uZWApO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkaXNwbGF5TmFtZU9mTm9kZShub2RlKSB7XG4gIGlmICghbm9kZSkgcmV0dXJuIG51bGw7XG5cbiAgY29uc3QgeyB0eXBlIH0gPSBub2RlO1xuXG4gIGlmICghdHlwZSkgcmV0dXJuIG51bGw7XG5cbiAgcmV0dXJuIHR5cGUuZGlzcGxheU5hbWUgfHwgKHR5cGVvZiB0eXBlID09PSAnZnVuY3Rpb24nID8gZnVuY3Rpb25OYW1lKHR5cGUpIDogdHlwZS5uYW1lIHx8IHR5cGUpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbm9kZVR5cGVGcm9tVHlwZSh0eXBlKSB7XG4gIGlmICh0eXBlb2YgdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICByZXR1cm4gJ2hvc3QnO1xuICB9XG4gIGlmICh0eXBlICYmIHR5cGUucHJvdG90eXBlICYmIHR5cGUucHJvdG90eXBlLmlzUmVhY3RDb21wb25lbnQpIHtcbiAgICByZXR1cm4gJ2NsYXNzJztcbiAgfVxuICByZXR1cm4gJ2Z1bmN0aW9uJztcbn1cblxuZnVuY3Rpb24gZ2V0SXRlcmF0b3JGbihvYmopIHtcbiAgY29uc3QgaXRlcmF0b3JGbiA9IG9iaiAmJiAoXG4gICAgKHR5cGVvZiBTeW1ib2wgPT09ICdmdW5jdGlvbicgJiYgdHlwZW9mIFN5bWJvbC5pdGVyYXRvciA9PT0gJ3N5bWJvbCcgJiYgb2JqW1N5bWJvbC5pdGVyYXRvcl0pXG4gICAgfHwgb2JqWydAQGl0ZXJhdG9yJ11cbiAgKTtcblxuICBpZiAodHlwZW9mIGl0ZXJhdG9yRm4gPT09ICdmdW5jdGlvbicpIHtcbiAgICByZXR1cm4gaXRlcmF0b3JGbjtcbiAgfVxuXG4gIHJldHVybiB1bmRlZmluZWQ7XG59XG5cbmZ1bmN0aW9uIGlzSXRlcmFibGUob2JqKSB7XG4gIHJldHVybiAhIWdldEl0ZXJhdG9yRm4ob2JqKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGlzQXJyYXlMaWtlKG9iaikge1xuICByZXR1cm4gQXJyYXkuaXNBcnJheShvYmopIHx8ICh0eXBlb2Ygb2JqICE9PSAnc3RyaW5nJyAmJiBpc0l0ZXJhYmxlKG9iaikpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZmxhdHRlbihhcnJzKSB7XG4gIC8vIG9wdGltaXplIGZvciB0aGUgbW9zdCBjb21tb24gY2FzZVxuICBpZiAoQXJyYXkuaXNBcnJheShhcnJzKSkge1xuICAgIHJldHVybiBhcnJzLnJlZHVjZShcbiAgICAgIChmbGF0QXJycywgaXRlbSkgPT4gZmxhdEFycnMuY29uY2F0KGlzQXJyYXlMaWtlKGl0ZW0pID8gZmxhdHRlbihpdGVtKSA6IGl0ZW0pLFxuICAgICAgW10sXG4gICAgKTtcbiAgfVxuXG4gIC8vIGZhbGxiYWNrIGZvciBhcmJpdHJhcnkgaXRlcmFibGUgY2hpbGRyZW5cbiAgbGV0IGZsYXRBcnJzID0gW107XG5cbiAgY29uc3QgaXRlcmF0b3JGbiA9IGdldEl0ZXJhdG9yRm4oYXJycyk7XG4gIGNvbnN0IGl0ZXJhdG9yID0gaXRlcmF0b3JGbi5jYWxsKGFycnMpO1xuXG4gIGxldCBzdGVwID0gaXRlcmF0b3IubmV4dCgpO1xuXG4gIHdoaWxlICghc3RlcC5kb25lKSB7XG4gICAgY29uc3QgaXRlbSA9IHN0ZXAudmFsdWU7XG4gICAgbGV0IGZsYXRJdGVtO1xuXG4gICAgaWYgKGlzQXJyYXlMaWtlKGl0ZW0pKSB7XG4gICAgICBmbGF0SXRlbSA9IGZsYXR0ZW4oaXRlbSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGZsYXRJdGVtID0gaXRlbTtcbiAgICB9XG5cbiAgICBmbGF0QXJycyA9IGZsYXRBcnJzLmNvbmNhdChmbGF0SXRlbSk7XG5cbiAgICBzdGVwID0gaXRlcmF0b3IubmV4dCgpO1xuICB9XG5cbiAgcmV0dXJuIGZsYXRBcnJzO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZW5zdXJlS2V5T3JVbmRlZmluZWQoa2V5KSB7XG4gIHJldHVybiBrZXkgfHwgKGtleSA9PT0gJycgPyAnJyA6IHVuZGVmaW5lZCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBlbGVtZW50VG9UcmVlKGVsLCByZWN1cnNlID0gZWxlbWVudFRvVHJlZSkge1xuICBpZiAodHlwZW9mIHJlY3Vyc2UgIT09ICdmdW5jdGlvbicgJiYgYXJndW1lbnRzLmxlbmd0aCA9PT0gMykge1xuICAgIC8vIHNwZWNpYWwgY2FzZSBmb3IgYmFja3dhcmRzIGNvbXBhdCBmb3IgYC5tYXAoZWxlbWVudFRvVHJlZSlgXG4gICAgcmVjdXJzZSA9IGVsZW1lbnRUb1RyZWU7IC8vIGVzbGludC1kaXNhYmxlLWxpbmUgbm8tcGFyYW0tcmVhc3NpZ25cbiAgfVxuICBpZiAoZWwgPT09IG51bGwgfHwgdHlwZW9mIGVsICE9PSAnb2JqZWN0JyB8fCAhKCd0eXBlJyBpbiBlbCkpIHtcbiAgICByZXR1cm4gZWw7XG4gIH1cbiAgY29uc3Qge1xuICAgIHR5cGUsXG4gICAgcHJvcHMsXG4gICAga2V5LFxuICAgIHJlZixcbiAgfSA9IGVsO1xuICBjb25zdCB7IGNoaWxkcmVuIH0gPSBwcm9wcztcbiAgbGV0IHJlbmRlcmVkID0gbnVsbDtcbiAgaWYgKGlzQXJyYXlMaWtlKGNoaWxkcmVuKSkge1xuICAgIHJlbmRlcmVkID0gZmxhdHRlbihjaGlsZHJlbikubWFwKCh4KSA9PiByZWN1cnNlKHgpKTtcbiAgfSBlbHNlIGlmICh0eXBlb2YgY2hpbGRyZW4gIT09ICd1bmRlZmluZWQnKSB7XG4gICAgcmVuZGVyZWQgPSByZWN1cnNlKGNoaWxkcmVuKTtcbiAgfVxuXG4gIGNvbnN0IG5vZGVUeXBlID0gbm9kZVR5cGVGcm9tVHlwZSh0eXBlKTtcblxuICBpZiAobm9kZVR5cGUgPT09ICdob3N0JyAmJiBwcm9wcy5kYW5nZXJvdXNseVNldElubmVySFRNTCkge1xuICAgIGlmIChwcm9wcy5jaGlsZHJlbiAhPSBudWxsKSB7XG4gICAgICBjb25zdCBlcnJvciA9IG5ldyBFcnJvcignQ2FuIG9ubHkgc2V0IG9uZSBvZiBgY2hpbGRyZW5gIG9yIGBwcm9wcy5kYW5nZXJvdXNseVNldElubmVySFRNTGAuJyk7XG4gICAgICBlcnJvci5uYW1lID0gJ0ludmFyaWFudCBWaW9sYXRpb24nO1xuICAgICAgdGhyb3cgZXJyb3I7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBub2RlVHlwZSxcbiAgICB0eXBlLFxuICAgIHByb3BzLFxuICAgIGtleTogZW5zdXJlS2V5T3JVbmRlZmluZWQoa2V5KSxcbiAgICByZWYsXG4gICAgaW5zdGFuY2U6IG51bGwsXG4gICAgcmVuZGVyZWQsXG4gIH07XG59XG5cbmZ1bmN0aW9uIG1hcEZpbmQoYXJyYXlsaWtlLCBtYXBwZXIsIGZpbmRlcikge1xuICBsZXQgZm91bmQ7XG4gIGNvbnN0IGlzRm91bmQgPSBBcnJheS5wcm90b3R5cGUuZmluZC5jYWxsKGFycmF5bGlrZSwgKGl0ZW0pID0+IHtcbiAgICBmb3VuZCA9IG1hcHBlcihpdGVtKTtcbiAgICByZXR1cm4gZmluZGVyKGZvdW5kKTtcbiAgfSk7XG4gIHJldHVybiBpc0ZvdW5kID8gZm91bmQgOiB1bmRlZmluZWQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBmaW5kRWxlbWVudChlbCwgcHJlZGljYXRlKSB7XG4gIGlmIChlbCA9PT0gbnVsbCB8fCB0eXBlb2YgZWwgIT09ICdvYmplY3QnIHx8ICEoJ3R5cGUnIGluIGVsKSkge1xuICAgIHJldHVybiB1bmRlZmluZWQ7XG4gIH1cbiAgaWYgKHByZWRpY2F0ZShlbCkpIHtcbiAgICByZXR1cm4gZWw7XG4gIH1cbiAgY29uc3QgeyByZW5kZXJlZCB9ID0gZWw7XG4gIGlmIChpc0FycmF5TGlrZShyZW5kZXJlZCkpIHtcbiAgICByZXR1cm4gbWFwRmluZChyZW5kZXJlZCwgKHgpID0+IGZpbmRFbGVtZW50KHgsIHByZWRpY2F0ZSksICh4KSA9PiB0eXBlb2YgeCAhPT0gJ3VuZGVmaW5lZCcpO1xuICB9XG4gIHJldHVybiBmaW5kRWxlbWVudChyZW5kZXJlZCwgcHJlZGljYXRlKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHByb3BzV2l0aEtleXNBbmRSZWYobm9kZSkge1xuICBpZiAobm9kZS5yZWYgIT09IG51bGwgfHwgbm9kZS5rZXkgIT09IG51bGwpIHtcbiAgICByZXR1cm4ge1xuICAgICAgLi4ubm9kZS5wcm9wcyxcbiAgICAgIGtleTogbm9kZS5rZXksXG4gICAgICByZWY6IG5vZGUucmVmLFxuICAgIH07XG4gIH1cbiAgcmV0dXJuIG5vZGUucHJvcHM7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRDb21wb25lbnRTdGFjayhcbiAgaGllcmFyY2h5LFxuICBnZXROb2RlVHlwZSA9IG5vZGVUeXBlRnJvbVR5cGUsXG4gIGdldERpc3BsYXlOYW1lID0gZGlzcGxheU5hbWVPZk5vZGUsXG4pIHtcbiAgY29uc3QgdHVwbGVzID0gaGllcmFyY2h5LmZpbHRlcigobm9kZSkgPT4gbm9kZS50eXBlICE9PSBSb290RmluZGVyKS5tYXAoKHgpID0+IFtcbiAgICBnZXROb2RlVHlwZSh4LnR5cGUpLFxuICAgIGdldERpc3BsYXlOYW1lKHgpLFxuICBdKS5jb25jYXQoW1tcbiAgICAnY2xhc3MnLFxuICAgICdXcmFwcGVyQ29tcG9uZW50JyxcbiAgXV0pO1xuXG4gIHJldHVybiB0dXBsZXMubWFwKChbLCBuYW1lXSwgaSwgYXJyKSA9PiB7XG4gICAgY29uc3QgWywgY2xvc2VzdENvbXBvbmVudF0gPSBhcnIuc2xpY2UoaSArIDEpLmZpbmQoKFtub2RlVHlwZV0pID0+IG5vZGVUeXBlICE9PSAnaG9zdCcpIHx8IFtdO1xuICAgIHJldHVybiBgXFxuICAgIGluICR7bmFtZX0ke2Nsb3Nlc3RDb21wb25lbnQgPyBgIChjcmVhdGVkIGJ5ICR7Y2xvc2VzdENvbXBvbmVudH0pYCA6ICcnfWA7XG4gIH0pLmpvaW4oJycpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gc2ltdWxhdGVFcnJvcihcbiAgZXJyb3IsXG4gIGNhdGNoaW5nSW5zdGFuY2UsXG4gIHJvb3ROb2RlLCAvLyBUT0RPOiByZW1vdmUgYHJvb3ROb2RlYCBuZXh0IHNlbXZlci1tYWpvclxuICBoaWVyYXJjaHksXG4gIGdldE5vZGVUeXBlID0gbm9kZVR5cGVGcm9tVHlwZSxcbiAgZ2V0RGlzcGxheU5hbWUgPSBkaXNwbGF5TmFtZU9mTm9kZSxcbiAgY2F0Y2hpbmdUeXBlID0ge30sXG4pIHtcbiAgY29uc3QgaW5zdGFuY2UgPSBjYXRjaGluZ0luc3RhbmNlIHx8IHt9O1xuXG4gIGNvbnN0IHsgY29tcG9uZW50RGlkQ2F0Y2ggfSA9IGluc3RhbmNlO1xuXG4gIGNvbnN0IHsgZ2V0RGVyaXZlZFN0YXRlRnJvbUVycm9yIH0gPSBjYXRjaGluZ1R5cGU7XG5cbiAgaWYgKCFjb21wb25lbnREaWRDYXRjaCAmJiAhZ2V0RGVyaXZlZFN0YXRlRnJvbUVycm9yKSB7XG4gICAgdGhyb3cgZXJyb3I7XG4gIH1cblxuICBpZiAoZ2V0RGVyaXZlZFN0YXRlRnJvbUVycm9yKSB7XG4gICAgY29uc3Qgc3RhdGVVcGRhdGUgPSBnZXREZXJpdmVkU3RhdGVGcm9tRXJyb3IuY2FsbChjYXRjaGluZ1R5cGUsIGVycm9yKTtcbiAgICBpbnN0YW5jZS5zZXRTdGF0ZShzdGF0ZVVwZGF0ZSk7XG4gIH1cblxuICBpZiAoY29tcG9uZW50RGlkQ2F0Y2gpIHtcbiAgICBjb25zdCBjb21wb25lbnRTdGFjayA9IGdldENvbXBvbmVudFN0YWNrKGhpZXJhcmNoeSwgZ2V0Tm9kZVR5cGUsIGdldERpc3BsYXlOYW1lKTtcbiAgICBjb21wb25lbnREaWRDYXRjaC5jYWxsKGluc3RhbmNlLCBlcnJvciwgeyBjb21wb25lbnRTdGFjayB9KTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0TWFza2VkQ29udGV4dChjb250ZXh0VHlwZXMsIHVubWFza2VkQ29udGV4dCkge1xuICBpZiAoIWNvbnRleHRUeXBlcyB8fCAhdW5tYXNrZWRDb250ZXh0KSB7XG4gICAgcmV0dXJuIHt9O1xuICB9XG4gIHJldHVybiBmcm9tRW50cmllcyhPYmplY3Qua2V5cyhjb250ZXh0VHlwZXMpLm1hcCgoa2V5KSA9PiBba2V5LCB1bm1hc2tlZENvbnRleHRba2V5XV0pKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldE5vZGVGcm9tUm9vdEZpbmRlcihpc0N1c3RvbUNvbXBvbmVudCwgdHJlZSwgb3B0aW9ucykge1xuICBpZiAoIWlzQ3VzdG9tQ29tcG9uZW50KG9wdGlvbnMud3JhcHBpbmdDb21wb25lbnQpKSB7XG4gICAgcmV0dXJuIHRyZWUucmVuZGVyZWQ7XG4gIH1cbiAgY29uc3Qgcm9vdEZpbmRlciA9IGZpbmRFbGVtZW50KHRyZWUsIChub2RlKSA9PiBub2RlLnR5cGUgPT09IFJvb3RGaW5kZXIpO1xuICBpZiAoIXJvb3RGaW5kZXIpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ2B3cmFwcGluZ0NvbXBvbmVudGAgbXVzdCByZW5kZXIgaXRzIGNoaWxkcmVuIScpO1xuICB9XG4gIHJldHVybiByb290RmluZGVyLnJlbmRlcmVkO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gd3JhcFdpdGhXcmFwcGluZ0NvbXBvbmVudChjcmVhdGVFbGVtZW50LCBub2RlLCBvcHRpb25zKSB7XG4gIGNvbnN0IHsgd3JhcHBpbmdDb21wb25lbnQsIHdyYXBwaW5nQ29tcG9uZW50UHJvcHMgfSA9IG9wdGlvbnM7XG4gIGlmICghd3JhcHBpbmdDb21wb25lbnQpIHtcbiAgICByZXR1cm4gbm9kZTtcbiAgfVxuICByZXR1cm4gY3JlYXRlRWxlbWVudChcbiAgICB3cmFwcGluZ0NvbXBvbmVudCxcbiAgICB3cmFwcGluZ0NvbXBvbmVudFByb3BzLFxuICAgIGNyZWF0ZUVsZW1lbnQoUm9vdEZpbmRlciwgbnVsbCwgbm9kZSksXG4gICk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRXcmFwcGluZ0NvbXBvbmVudE1vdW50UmVuZGVyZXIoeyB0b1RyZWUsIGdldE1vdW50V3JhcHBlckluc3RhbmNlIH0pIHtcbiAgcmV0dXJuIHtcbiAgICBnZXROb2RlKCkge1xuICAgICAgY29uc3QgaW5zdGFuY2UgPSBnZXRNb3VudFdyYXBwZXJJbnN0YW5jZSgpO1xuICAgICAgcmV0dXJuIGluc3RhbmNlID8gdG9UcmVlKGluc3RhbmNlKS5yZW5kZXJlZCA6IG51bGw7XG4gICAgfSxcbiAgICByZW5kZXIoZWwsIGNvbnRleHQsIGNhbGxiYWNrKSB7XG4gICAgICBjb25zdCBpbnN0YW5jZSA9IGdldE1vdW50V3JhcHBlckluc3RhbmNlKCk7XG4gICAgICBpZiAoIWluc3RhbmNlKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcignVGhlIHdyYXBwaW5nIGNvbXBvbmVudCBtYXkgbm90IGJlIHVwZGF0ZWQgaWYgdGhlIHJvb3QgaXMgdW5tb3VudGVkLicpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIGluc3RhbmNlLnNldFdyYXBwaW5nQ29tcG9uZW50UHJvcHMoZWwucHJvcHMsIGNhbGxiYWNrKTtcbiAgICB9LFxuICB9O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZmFrZUR5bmFtaWNJbXBvcnQobW9kdWxlVG9JbXBvcnQpIHtcbiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZSh7IGRlZmF1bHQ6IG1vZHVsZVRvSW1wb3J0IH0pO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcGFyZU5vZGVUeXBlT2Yobm9kZSwgbWF0Y2hpbmdUeXBlT2YpIHtcbiAgaWYgKCFub2RlKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIHJldHVybiBub2RlLiQkdHlwZW9mID09PSBtYXRjaGluZ1R5cGVPZjtcbn1cblxuLy8gVE9ETzogd2hlbiBlbnp5bWUgdjMuMTIuMCBpcyByZXF1aXJlZCwgZGVsZXRlIHRoaXNcbmV4cG9ydCBmdW5jdGlvbiBzcHlNZXRob2QoaW5zdGFuY2UsIG1ldGhvZE5hbWUsIGdldFN0dWIgPSAoKSA9PiB7fSkge1xuICBsZXQgbGFzdFJldHVyblZhbHVlO1xuICBjb25zdCBvcmlnaW5hbE1ldGhvZCA9IGluc3RhbmNlW21ldGhvZE5hbWVdO1xuICBjb25zdCBoYXNPd24gPSBoYXMoaW5zdGFuY2UsIG1ldGhvZE5hbWUpO1xuICBsZXQgZGVzY3JpcHRvcjtcbiAgaWYgKGhhc093bikge1xuICAgIGRlc2NyaXB0b3IgPSBPYmplY3QuZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKGluc3RhbmNlLCBtZXRob2ROYW1lKTtcbiAgfVxuICBPYmplY3QuZGVmaW5lUHJvcGVydHkoaW5zdGFuY2UsIG1ldGhvZE5hbWUsIHtcbiAgICBjb25maWd1cmFibGU6IHRydWUsXG4gICAgZW51bWVyYWJsZTogIWRlc2NyaXB0b3IgfHwgISFkZXNjcmlwdG9yLmVudW1lcmFibGUsXG4gICAgdmFsdWU6IGdldFN0dWIob3JpZ2luYWxNZXRob2QpIHx8IGZ1bmN0aW9uIHNwaWVkKC4uLmFyZ3MpIHtcbiAgICAgIGNvbnN0IHJlc3VsdCA9IG9yaWdpbmFsTWV0aG9kLmFwcGx5KHRoaXMsIGFyZ3MpO1xuICAgICAgbGFzdFJldHVyblZhbHVlID0gcmVzdWx0O1xuICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICB9LFxuICB9KTtcbiAgcmV0dXJuIHtcbiAgICByZXN0b3JlKCkge1xuICAgICAgaWYgKGhhc093bikge1xuICAgICAgICBpZiAoZGVzY3JpcHRvcikge1xuICAgICAgICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eShpbnN0YW5jZSwgbWV0aG9kTmFtZSwgZGVzY3JpcHRvcik7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgLyogZXNsaW50LWRpc2FibGUgbm8tcGFyYW0tcmVhc3NpZ24gKi9cbiAgICAgICAgICBpbnN0YW5jZVttZXRob2ROYW1lXSA9IG9yaWdpbmFsTWV0aG9kO1xuICAgICAgICAgIC8qIGVzbGludC1lbmFibGUgbm8tcGFyYW0tcmVhc3NpZ24gKi9cbiAgICAgICAgfVxuICAgICAgfSBlbHNlIHtcbiAgICAgICAgLyogZXNsaW50LWRpc2FibGUgbm8tcGFyYW0tcmVhc3NpZ24gKi9cbiAgICAgICAgZGVsZXRlIGluc3RhbmNlW21ldGhvZE5hbWVdO1xuICAgICAgICAvKiBlc2xpbnQtZW5hYmxlIG5vLXBhcmFtLXJlYXNzaWduICovXG4gICAgICB9XG4gICAgfSxcbiAgICBnZXRMYXN0UmV0dXJuVmFsdWUoKSB7XG4gICAgICByZXR1cm4gbGFzdFJldHVyblZhbHVlO1xuICAgIH0sXG4gIH07XG59XG5cbi8vIFRPRE86IHdoZW4gZW56eW1lIHYzLjEyLjAgaXMgcmVxdWlyZWQsIGRlbGV0ZSB0aGlzXG5leHBvcnQgZnVuY3Rpb24gc3B5UHJvcGVydHkoaW5zdGFuY2UsIHByb3BlcnR5TmFtZSwgaGFuZGxlcnMgPSB7fSkge1xuICBjb25zdCBvcmlnaW5hbFZhbHVlID0gaW5zdGFuY2VbcHJvcGVydHlOYW1lXTtcbiAgY29uc3QgaGFzT3duID0gaGFzKGluc3RhbmNlLCBwcm9wZXJ0eU5hbWUpO1xuICBsZXQgZGVzY3JpcHRvcjtcbiAgaWYgKGhhc093bikge1xuICAgIGRlc2NyaXB0b3IgPSBPYmplY3QuZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKGluc3RhbmNlLCBwcm9wZXJ0eU5hbWUpO1xuICB9XG4gIGxldCB3YXNBc3NpZ25lZCA9IGZhbHNlO1xuICBsZXQgaG9sZGVyID0gb3JpZ2luYWxWYWx1ZTtcbiAgY29uc3QgZ2V0ViA9IGhhbmRsZXJzLmdldCA/ICgpID0+IHtcbiAgICBjb25zdCB2YWx1ZSA9IGRlc2NyaXB0b3IgJiYgZGVzY3JpcHRvci5nZXQgPyBkZXNjcmlwdG9yLmdldC5jYWxsKGluc3RhbmNlKSA6IGhvbGRlcjtcbiAgICByZXR1cm4gaGFuZGxlcnMuZ2V0LmNhbGwoaW5zdGFuY2UsIHZhbHVlKTtcbiAgfSA6ICgpID0+IGhvbGRlcjtcbiAgY29uc3Qgc2V0ID0gaGFuZGxlcnMuc2V0ID8gKG5ld1ZhbHVlKSA9PiB7XG4gICAgd2FzQXNzaWduZWQgPSB0cnVlO1xuICAgIGNvbnN0IGhhbmRsZXJOZXdWYWx1ZSA9IGhhbmRsZXJzLnNldC5jYWxsKGluc3RhbmNlLCBob2xkZXIsIG5ld1ZhbHVlKTtcbiAgICBob2xkZXIgPSBoYW5kbGVyTmV3VmFsdWU7XG4gICAgaWYgKGRlc2NyaXB0b3IgJiYgZGVzY3JpcHRvci5zZXQpIHtcbiAgICAgIGRlc2NyaXB0b3Iuc2V0LmNhbGwoaW5zdGFuY2UsIGhvbGRlcik7XG4gICAgfVxuICB9IDogKHYpID0+IHtcbiAgICB3YXNBc3NpZ25lZCA9IHRydWU7XG4gICAgaG9sZGVyID0gdjtcbiAgfTtcbiAgT2JqZWN0LmRlZmluZVByb3BlcnR5KGluc3RhbmNlLCBwcm9wZXJ0eU5hbWUsIHtcbiAgICBjb25maWd1cmFibGU6IHRydWUsXG4gICAgZW51bWVyYWJsZTogIWRlc2NyaXB0b3IgfHwgISFkZXNjcmlwdG9yLmVudW1lcmFibGUsXG4gICAgZ2V0OiBnZXRWLFxuICAgIHNldCxcbiAgfSk7XG5cbiAgcmV0dXJuIHtcbiAgICByZXN0b3JlKCkge1xuICAgICAgaWYgKGhhc093bikge1xuICAgICAgICBpZiAoZGVzY3JpcHRvcikge1xuICAgICAgICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eShpbnN0YW5jZSwgcHJvcGVydHlOYW1lLCBkZXNjcmlwdG9yKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAvKiBlc2xpbnQtZGlzYWJsZSBuby1wYXJhbS1yZWFzc2lnbiAqL1xuICAgICAgICAgIGluc3RhbmNlW3Byb3BlcnR5TmFtZV0gPSBob2xkZXI7XG4gICAgICAgICAgLyogZXNsaW50LWVuYWJsZSBuby1wYXJhbS1yZWFzc2lnbiAqL1xuICAgICAgICB9XG4gICAgICB9IGVsc2Uge1xuICAgICAgICAvKiBlc2xpbnQtZGlzYWJsZSBuby1wYXJhbS1yZWFzc2lnbiAqL1xuICAgICAgICBkZWxldGUgaW5zdGFuY2VbcHJvcGVydHlOYW1lXTtcbiAgICAgICAgLyogZXNsaW50LWVuYWJsZSBuby1wYXJhbS1yZWFzc2lnbiAqL1xuICAgICAgfVxuICAgIH0sXG4gICAgd2FzQXNzaWduZWQoKSB7XG4gICAgICByZXR1cm4gd2FzQXNzaWduZWQ7XG4gICAgfSxcbiAgfTtcbn1c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
//# sourceMappingURL=Utils.js.map