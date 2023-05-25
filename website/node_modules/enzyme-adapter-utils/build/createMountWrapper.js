"use strict";

function _typeof(obj) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, _typeof(obj); }
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = createMountWrapper;
var _react = _interopRequireDefault(require("react"));
var _propTypes = _interopRequireDefault(require("prop-types"));
var _airbnbPropTypes = require("airbnb-prop-types");
var _RootFinder = _interopRequireDefault(require("./RootFinder"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }
function _extends() { _extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }
function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ownKeys(Object(source), !0).forEach(function (key) { _defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) _setPrototypeOf(subClass, superClass); }
function _setPrototypeOf(o, p) { _setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return _setPrototypeOf(o, p); }
function _createSuper(Derived) { var hasNativeReflectConstruct = _isNativeReflectConstruct(); return function _createSuperInternal() { var Super = _getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = _getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return _possibleConstructorReturn(this, result); }; }
function _possibleConstructorReturn(self, call) { if (call && (_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return _assertThisInitialized(self); }
function _assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function _isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function _getPrototypeOf(o) { _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return _getPrototypeOf(o); }
/* eslint react/forbid-prop-types: 0 */

var stringOrFunction = _propTypes["default"].oneOfType([_propTypes["default"].func, _propTypes["default"].string]);
var makeValidElementType = function makeValidElementType(adapter) {
  if (!adapter) {
    return stringOrFunction;
  }
  function validElementTypeRequired(props, propName) {
    if (!adapter.isValidElementType) {
      for (var _len = arguments.length, args = new Array(_len > 2 ? _len - 2 : 0), _key = 2; _key < _len; _key++) {
        args[_key - 2] = arguments[_key];
      }
      return stringOrFunction.isRequired.apply(stringOrFunction, [props, propName].concat(args));
    }
    var propValue = props[propName]; // eslint-disable-line react/destructuring-assignment
    if (adapter.isValidElementType(propValue)) {
      return null;
    }
    return new TypeError("".concat(propName, " must be a valid element type!"));
  }
  function validElementType(props, propName) {
    var propValue = props[propName];
    if (propValue == null) {
      return null;
    }
    for (var _len2 = arguments.length, args = new Array(_len2 > 2 ? _len2 - 2 : 0), _key2 = 2; _key2 < _len2; _key2++) {
      args[_key2 - 2] = arguments[_key2];
    }
    return validElementTypeRequired.apply(void 0, [props, propName].concat(args));
  }
  validElementType.isRequired = validElementTypeRequired;
  return validElementType;
};

/**
 * This is a utility component to wrap around the nodes we are
 * passing in to `mount()`. Theoretically, you could do everything
 * we are doing without this, but this makes it easier since
 * `renderIntoDocument()` doesn't really pass back a reference to
 * the DOM node it rendered to, so we can't really "re-render" to
 * pass new props in.
 */
function createMountWrapper(node) {
  var options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
  var adapter = options.adapter,
    WrappingComponent = options.wrappingComponent;
  var WrapperComponent = /*#__PURE__*/function (_React$Component) {
    _inherits(WrapperComponent, _React$Component);
    var _super = _createSuper(WrapperComponent);
    function WrapperComponent() {
      var _this;
      _classCallCheck(this, WrapperComponent);
      for (var _len3 = arguments.length, args = new Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {
        args[_key3] = arguments[_key3];
      }
      _this = _super.call.apply(_super, [this].concat(args));
      var _this$props = _this.props,
        props = _this$props.props,
        wrappingComponentProps = _this$props.wrappingComponentProps,
        context = _this$props.context;
      _this.state = {
        mount: true,
        props: props,
        wrappingComponentProps: wrappingComponentProps,
        context: context
      };
      return _this;
    }

    // eslint-disable-next-line react/no-unused-class-component-methods
    _createClass(WrapperComponent, [{
      key: "setChildProps",
      value: function setChildProps(newProps, newContext) {
        var callback = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : undefined;
        var _this$state = this.state,
          oldProps = _this$state.props,
          oldContext = _this$state.context;
        var props = _objectSpread(_objectSpread({}, oldProps), newProps);
        var context = _objectSpread(_objectSpread({}, oldContext), newContext);
        this.setState({
          props: props,
          context: context
        }, callback);
      }

      // eslint-disable-next-line react/no-unused-class-component-methods
    }, {
      key: "setWrappingComponentProps",
      value: function setWrappingComponentProps(props) {
        var callback = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : undefined;
        this.setState({
          wrappingComponentProps: props
        }, callback);
      }
    }, {
      key: "render",
      value: function render() {
        var _this$props2 = this.props,
          Component = _this$props2.Component,
          refProp = _this$props2.refProp;
        var _this$state2 = this.state,
          mount = _this$state2.mount,
          props = _this$state2.props,
          wrappingComponentProps = _this$state2.wrappingComponentProps;
        if (!mount) return null;
        // eslint-disable-next-line react/jsx-props-no-spreading
        var component = /*#__PURE__*/_react["default"].createElement(Component, _extends({
          ref: refProp
        }, props));
        if (WrappingComponent) {
          return (
            /*#__PURE__*/
            // eslint-disable-next-line react/jsx-props-no-spreading
            _react["default"].createElement(WrappingComponent, wrappingComponentProps, /*#__PURE__*/_react["default"].createElement(_RootFinder["default"], null, component))
          );
        }
        return component;
      }
    }]);
    return WrapperComponent;
  }(_react["default"].Component);
  WrapperComponent.propTypes = {
    Component: makeValidElementType(adapter).isRequired,
    refProp: _propTypes["default"].oneOfType([_propTypes["default"].string, (0, _airbnbPropTypes.ref)()]),
    props: _propTypes["default"].object.isRequired,
    wrappingComponentProps: _propTypes["default"].object,
    context: _propTypes["default"].object
  };
  WrapperComponent.defaultProps = {
    refProp: null,
    context: null,
    wrappingComponentProps: null
  };
  if (options.context && (node.type.contextTypes || options.childContextTypes)) {
    // For full rendering, we are using this wrapper component to provide context if it is
    // specified in both the options AND the child component defines `contextTypes` statically
    // OR the merged context types for all children (the node component or deeper children) are
    // specified in options parameter under childContextTypes.
    // In that case, we define both a `getChildContext()` function and a `childContextTypes` prop.
    var childContextTypes = _objectSpread(_objectSpread({}, node.type.contextTypes), options.childContextTypes);
    WrapperComponent.prototype.getChildContext = function getChildContext() {
      return this.state.context;
    };
    WrapperComponent.childContextTypes = childContextTypes;
  }
  return WrapperComponent;
}
module.exports = exports.default;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJuYW1lcyI6WyJzdHJpbmdPckZ1bmN0aW9uIiwiUHJvcFR5cGVzIiwib25lT2ZUeXBlIiwiZnVuYyIsInN0cmluZyIsIm1ha2VWYWxpZEVsZW1lbnRUeXBlIiwiYWRhcHRlciIsInZhbGlkRWxlbWVudFR5cGVSZXF1aXJlZCIsInByb3BzIiwicHJvcE5hbWUiLCJpc1ZhbGlkRWxlbWVudFR5cGUiLCJhcmdzIiwiaXNSZXF1aXJlZCIsInByb3BWYWx1ZSIsIlR5cGVFcnJvciIsInZhbGlkRWxlbWVudFR5cGUiLCJjcmVhdGVNb3VudFdyYXBwZXIiLCJub2RlIiwib3B0aW9ucyIsIldyYXBwaW5nQ29tcG9uZW50Iiwid3JhcHBpbmdDb21wb25lbnQiLCJXcmFwcGVyQ29tcG9uZW50Iiwid3JhcHBpbmdDb21wb25lbnRQcm9wcyIsImNvbnRleHQiLCJzdGF0ZSIsIm1vdW50IiwibmV3UHJvcHMiLCJuZXdDb250ZXh0IiwiY2FsbGJhY2siLCJ1bmRlZmluZWQiLCJvbGRQcm9wcyIsIm9sZENvbnRleHQiLCJzZXRTdGF0ZSIsIkNvbXBvbmVudCIsInJlZlByb3AiLCJjb21wb25lbnQiLCJSZWFjdCIsInByb3BUeXBlcyIsInJlZiIsIm9iamVjdCIsImRlZmF1bHRQcm9wcyIsInR5cGUiLCJjb250ZXh0VHlwZXMiLCJjaGlsZENvbnRleHRUeXBlcyIsInByb3RvdHlwZSIsImdldENoaWxkQ29udGV4dCJdLCJzb3VyY2VzIjpbIi4uL3NyYy9jcmVhdGVNb3VudFdyYXBwZXIuanN4Il0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCBmcm9tICdyZWFjdCc7XG5pbXBvcnQgUHJvcFR5cGVzIGZyb20gJ3Byb3AtdHlwZXMnO1xuaW1wb3J0IHsgcmVmIH0gZnJvbSAnYWlyYm5iLXByb3AtdHlwZXMnO1xuaW1wb3J0IFJvb3RGaW5kZXIgZnJvbSAnLi9Sb290RmluZGVyJztcblxuLyogZXNsaW50IHJlYWN0L2ZvcmJpZC1wcm9wLXR5cGVzOiAwICovXG5cbmNvbnN0IHN0cmluZ09yRnVuY3Rpb24gPSBQcm9wVHlwZXMub25lT2ZUeXBlKFtQcm9wVHlwZXMuZnVuYywgUHJvcFR5cGVzLnN0cmluZ10pO1xuY29uc3QgbWFrZVZhbGlkRWxlbWVudFR5cGUgPSAoYWRhcHRlcikgPT4ge1xuICBpZiAoIWFkYXB0ZXIpIHtcbiAgICByZXR1cm4gc3RyaW5nT3JGdW5jdGlvbjtcbiAgfVxuXG4gIGZ1bmN0aW9uIHZhbGlkRWxlbWVudFR5cGVSZXF1aXJlZChwcm9wcywgcHJvcE5hbWUsIC4uLmFyZ3MpIHtcbiAgICBpZiAoIWFkYXB0ZXIuaXNWYWxpZEVsZW1lbnRUeXBlKSB7XG4gICAgICByZXR1cm4gc3RyaW5nT3JGdW5jdGlvbi5pc1JlcXVpcmVkKHByb3BzLCBwcm9wTmFtZSwgLi4uYXJncyk7XG4gICAgfVxuICAgIGNvbnN0IHByb3BWYWx1ZSA9IHByb3BzW3Byb3BOYW1lXTsgLy8gZXNsaW50LWRpc2FibGUtbGluZSByZWFjdC9kZXN0cnVjdHVyaW5nLWFzc2lnbm1lbnRcbiAgICBpZiAoYWRhcHRlci5pc1ZhbGlkRWxlbWVudFR5cGUocHJvcFZhbHVlKSkge1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIHJldHVybiBuZXcgVHlwZUVycm9yKGAke3Byb3BOYW1lfSBtdXN0IGJlIGEgdmFsaWQgZWxlbWVudCB0eXBlIWApO1xuICB9XG5cbiAgZnVuY3Rpb24gdmFsaWRFbGVtZW50VHlwZShwcm9wcywgcHJvcE5hbWUsIC4uLmFyZ3MpIHtcbiAgICBjb25zdCBwcm9wVmFsdWUgPSBwcm9wc1twcm9wTmFtZV07XG4gICAgaWYgKHByb3BWYWx1ZSA9PSBudWxsKSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gICAgcmV0dXJuIHZhbGlkRWxlbWVudFR5cGVSZXF1aXJlZChwcm9wcywgcHJvcE5hbWUsIC4uLmFyZ3MpO1xuICB9XG4gIHZhbGlkRWxlbWVudFR5cGUuaXNSZXF1aXJlZCA9IHZhbGlkRWxlbWVudFR5cGVSZXF1aXJlZDtcblxuICByZXR1cm4gdmFsaWRFbGVtZW50VHlwZTtcbn07XG5cbi8qKlxuICogVGhpcyBpcyBhIHV0aWxpdHkgY29tcG9uZW50IHRvIHdyYXAgYXJvdW5kIHRoZSBub2RlcyB3ZSBhcmVcbiAqIHBhc3NpbmcgaW4gdG8gYG1vdW50KClgLiBUaGVvcmV0aWNhbGx5LCB5b3UgY291bGQgZG8gZXZlcnl0aGluZ1xuICogd2UgYXJlIGRvaW5nIHdpdGhvdXQgdGhpcywgYnV0IHRoaXMgbWFrZXMgaXQgZWFzaWVyIHNpbmNlXG4gKiBgcmVuZGVySW50b0RvY3VtZW50KClgIGRvZXNuJ3QgcmVhbGx5IHBhc3MgYmFjayBhIHJlZmVyZW5jZSB0b1xuICogdGhlIERPTSBub2RlIGl0IHJlbmRlcmVkIHRvLCBzbyB3ZSBjYW4ndCByZWFsbHkgXCJyZS1yZW5kZXJcIiB0b1xuICogcGFzcyBuZXcgcHJvcHMgaW4uXG4gKi9cbmV4cG9ydCBkZWZhdWx0IGZ1bmN0aW9uIGNyZWF0ZU1vdW50V3JhcHBlcihub2RlLCBvcHRpb25zID0ge30pIHtcbiAgY29uc3QgeyBhZGFwdGVyLCB3cmFwcGluZ0NvbXBvbmVudDogV3JhcHBpbmdDb21wb25lbnQgfSA9IG9wdGlvbnM7XG5cbiAgY2xhc3MgV3JhcHBlckNvbXBvbmVudCBleHRlbmRzIFJlYWN0LkNvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IoLi4uYXJncykge1xuICAgICAgc3VwZXIoLi4uYXJncyk7XG4gICAgICBjb25zdCB7IHByb3BzLCB3cmFwcGluZ0NvbXBvbmVudFByb3BzLCBjb250ZXh0IH0gPSB0aGlzLnByb3BzO1xuICAgICAgdGhpcy5zdGF0ZSA9IHtcbiAgICAgICAgbW91bnQ6IHRydWUsXG4gICAgICAgIHByb3BzLFxuICAgICAgICB3cmFwcGluZ0NvbXBvbmVudFByb3BzLFxuICAgICAgICBjb250ZXh0LFxuICAgICAgfTtcbiAgICB9XG5cbiAgICAvLyBlc2xpbnQtZGlzYWJsZS1uZXh0LWxpbmUgcmVhY3Qvbm8tdW51c2VkLWNsYXNzLWNvbXBvbmVudC1tZXRob2RzXG4gICAgc2V0Q2hpbGRQcm9wcyhuZXdQcm9wcywgbmV3Q29udGV4dCwgY2FsbGJhY2sgPSB1bmRlZmluZWQpIHtcbiAgICAgIGNvbnN0IHsgcHJvcHM6IG9sZFByb3BzLCBjb250ZXh0OiBvbGRDb250ZXh0IH0gPSB0aGlzLnN0YXRlO1xuICAgICAgY29uc3QgcHJvcHMgPSB7IC4uLm9sZFByb3BzLCAuLi5uZXdQcm9wcyB9O1xuICAgICAgY29uc3QgY29udGV4dCA9IHsgLi4ub2xkQ29udGV4dCwgLi4ubmV3Q29udGV4dCB9O1xuICAgICAgdGhpcy5zZXRTdGF0ZSh7IHByb3BzLCBjb250ZXh0IH0sIGNhbGxiYWNrKTtcbiAgICB9XG5cbiAgICAvLyBlc2xpbnQtZGlzYWJsZS1uZXh0LWxpbmUgcmVhY3Qvbm8tdW51c2VkLWNsYXNzLWNvbXBvbmVudC1tZXRob2RzXG4gICAgc2V0V3JhcHBpbmdDb21wb25lbnRQcm9wcyhwcm9wcywgY2FsbGJhY2sgPSB1bmRlZmluZWQpIHtcbiAgICAgIHRoaXMuc2V0U3RhdGUoeyB3cmFwcGluZ0NvbXBvbmVudFByb3BzOiBwcm9wcyB9LCBjYWxsYmFjayk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCkge1xuICAgICAgY29uc3QgeyBDb21wb25lbnQsIHJlZlByb3AgfSA9IHRoaXMucHJvcHM7XG4gICAgICBjb25zdCB7IG1vdW50LCBwcm9wcywgd3JhcHBpbmdDb21wb25lbnRQcm9wcyB9ID0gdGhpcy5zdGF0ZTtcbiAgICAgIGlmICghbW91bnQpIHJldHVybiBudWxsO1xuICAgICAgLy8gZXNsaW50LWRpc2FibGUtbmV4dC1saW5lIHJlYWN0L2pzeC1wcm9wcy1uby1zcHJlYWRpbmdcbiAgICAgIGNvbnN0IGNvbXBvbmVudCA9IDxDb21wb25lbnQgcmVmPXtyZWZQcm9wfSB7Li4ucHJvcHN9IC8+O1xuICAgICAgaWYgKFdyYXBwaW5nQ29tcG9uZW50KSB7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgLy8gZXNsaW50LWRpc2FibGUtbmV4dC1saW5lIHJlYWN0L2pzeC1wcm9wcy1uby1zcHJlYWRpbmdcbiAgICAgICAgICA8V3JhcHBpbmdDb21wb25lbnQgey4uLndyYXBwaW5nQ29tcG9uZW50UHJvcHN9PlxuICAgICAgICAgICAgPFJvb3RGaW5kZXI+e2NvbXBvbmVudH08L1Jvb3RGaW5kZXI+XG4gICAgICAgICAgPC9XcmFwcGluZ0NvbXBvbmVudD5cbiAgICAgICAgKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBjb21wb25lbnQ7XG4gICAgfVxuICB9XG4gIFdyYXBwZXJDb21wb25lbnQucHJvcFR5cGVzID0ge1xuICAgIENvbXBvbmVudDogbWFrZVZhbGlkRWxlbWVudFR5cGUoYWRhcHRlcikuaXNSZXF1aXJlZCxcbiAgICByZWZQcm9wOiBQcm9wVHlwZXMub25lT2ZUeXBlKFtQcm9wVHlwZXMuc3RyaW5nLCByZWYoKV0pLFxuICAgIHByb3BzOiBQcm9wVHlwZXMub2JqZWN0LmlzUmVxdWlyZWQsXG4gICAgd3JhcHBpbmdDb21wb25lbnRQcm9wczogUHJvcFR5cGVzLm9iamVjdCxcbiAgICBjb250ZXh0OiBQcm9wVHlwZXMub2JqZWN0LFxuICB9O1xuICBXcmFwcGVyQ29tcG9uZW50LmRlZmF1bHRQcm9wcyA9IHtcbiAgICByZWZQcm9wOiBudWxsLFxuICAgIGNvbnRleHQ6IG51bGwsXG4gICAgd3JhcHBpbmdDb21wb25lbnRQcm9wczogbnVsbCxcbiAgfTtcblxuICBpZiAob3B0aW9ucy5jb250ZXh0ICYmIChub2RlLnR5cGUuY29udGV4dFR5cGVzIHx8IG9wdGlvbnMuY2hpbGRDb250ZXh0VHlwZXMpKSB7XG4gICAgLy8gRm9yIGZ1bGwgcmVuZGVyaW5nLCB3ZSBhcmUgdXNpbmcgdGhpcyB3cmFwcGVyIGNvbXBvbmVudCB0byBwcm92aWRlIGNvbnRleHQgaWYgaXQgaXNcbiAgICAvLyBzcGVjaWZpZWQgaW4gYm90aCB0aGUgb3B0aW9ucyBBTkQgdGhlIGNoaWxkIGNvbXBvbmVudCBkZWZpbmVzIGBjb250ZXh0VHlwZXNgIHN0YXRpY2FsbHlcbiAgICAvLyBPUiB0aGUgbWVyZ2VkIGNvbnRleHQgdHlwZXMgZm9yIGFsbCBjaGlsZHJlbiAodGhlIG5vZGUgY29tcG9uZW50IG9yIGRlZXBlciBjaGlsZHJlbikgYXJlXG4gICAgLy8gc3BlY2lmaWVkIGluIG9wdGlvbnMgcGFyYW1ldGVyIHVuZGVyIGNoaWxkQ29udGV4dFR5cGVzLlxuICAgIC8vIEluIHRoYXQgY2FzZSwgd2UgZGVmaW5lIGJvdGggYSBgZ2V0Q2hpbGRDb250ZXh0KClgIGZ1bmN0aW9uIGFuZCBhIGBjaGlsZENvbnRleHRUeXBlc2AgcHJvcC5cbiAgICBjb25zdCBjaGlsZENvbnRleHRUeXBlcyA9IHtcbiAgICAgIC4uLm5vZGUudHlwZS5jb250ZXh0VHlwZXMsXG4gICAgICAuLi5vcHRpb25zLmNoaWxkQ29udGV4dFR5cGVzLFxuICAgIH07XG5cbiAgICBXcmFwcGVyQ29tcG9uZW50LnByb3RvdHlwZS5nZXRDaGlsZENvbnRleHQgPSBmdW5jdGlvbiBnZXRDaGlsZENvbnRleHQoKSB7XG4gICAgICByZXR1cm4gdGhpcy5zdGF0ZS5jb250ZXh0O1xuICAgIH07XG4gICAgV3JhcHBlckNvbXBvbmVudC5jaGlsZENvbnRleHRUeXBlcyA9IGNoaWxkQ29udGV4dFR5cGVzO1xuICB9XG4gIHJldHVybiBXcmFwcGVyQ29tcG9uZW50O1xufVxuIl0sIm1hcHBpbmdzIjoiOzs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBQ0E7QUFBc0M7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBRXRDOztBQUVBLElBQU1BLGdCQUFnQixHQUFHQyxxQkFBUyxDQUFDQyxTQUFTLENBQUMsQ0FBQ0QscUJBQVMsQ0FBQ0UsSUFBSSxFQUFFRixxQkFBUyxDQUFDRyxNQUFNLENBQUMsQ0FBQztBQUNoRixJQUFNQyxvQkFBb0IsR0FBRyxTQUF2QkEsb0JBQW9CLENBQUlDLE9BQU8sRUFBSztFQUN4QyxJQUFJLENBQUNBLE9BQU8sRUFBRTtJQUNaLE9B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
//# sourceMappingURL=createMountWrapper.js.map