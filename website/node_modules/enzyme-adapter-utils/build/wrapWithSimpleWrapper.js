"use strict";

function _typeof(obj) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, _typeof(obj); }
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = wrap;
var _object = _interopRequireDefault(require("object.assign"));
var _react = _interopRequireDefault(require("react"));
var _semver = require("semver");
var _airbnbPropTypes = require("airbnb-prop-types");
var _propTypes = _interopRequireDefault(require("prop-types"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }
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
var propTypes = {
  children: (0, _airbnbPropTypes.or)([(0, _airbnbPropTypes.explicitNull)().isRequired, _propTypes["default"].node.isRequired])
};
var defaultProps = {
  children: undefined
};
var Wrapper = ((0, _semver.intersects)('>= 0.14', _react["default"].version)
// eslint-disable-next-line prefer-arrow-callback
? function () {
  return (0, _object["default"])(function SimpleSFCWrapper(_ref) {
    var children = _ref.children;
    return children;
  }, {
    propTypes: propTypes,
    defaultProps: defaultProps
  });
} : function () {
  var SimpleClassWrapper = /*#__PURE__*/function (_React$Component) {
    _inherits(SimpleClassWrapper, _React$Component);
    var _super = _createSuper(SimpleClassWrapper);
    function SimpleClassWrapper() {
      _classCallCheck(this, SimpleClassWrapper);
      return _super.apply(this, arguments);
    }
    _createClass(SimpleClassWrapper, [{
      key: "render",
      value: function render() {
        var children = this.props.children;
        return children;
      }
    }]);
    return SimpleClassWrapper;
  }(_react["default"].Component);
  SimpleClassWrapper.propTypes = propTypes;
  SimpleClassWrapper.defaultProps = defaultProps;
  return SimpleClassWrapper;
})();
function wrap(element) {
  return /*#__PURE__*/_react["default"].createElement(Wrapper, null, element);
}
module.exports = exports.default;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJuYW1lcyI6WyJwcm9wVHlwZXMiLCJjaGlsZHJlbiIsIm9yIiwiZXhwbGljaXROdWxsIiwiaXNSZXF1aXJlZCIsIlByb3BUeXBlcyIsIm5vZGUiLCJkZWZhdWx0UHJvcHMiLCJ1bmRlZmluZWQiLCJXcmFwcGVyIiwiaW50ZXJzZWN0cyIsIlJlYWN0IiwidmVyc2lvbiIsIlNpbXBsZVNGQ1dyYXBwZXIiLCJTaW1wbGVDbGFzc1dyYXBwZXIiLCJwcm9wcyIsIkNvbXBvbmVudCIsIndyYXAiLCJlbGVtZW50Il0sInNvdXJjZXMiOlsiLi4vc3JjL3dyYXBXaXRoU2ltcGxlV3JhcHBlci5qc3giXSwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IGludGVyc2VjdHMgfSBmcm9tICdzZW12ZXInO1xuaW1wb3J0IHsgb3IsIGV4cGxpY2l0TnVsbCB9IGZyb20gJ2FpcmJuYi1wcm9wLXR5cGVzJztcbmltcG9ydCBQcm9wVHlwZXMgZnJvbSAncHJvcC10eXBlcyc7XG5cbmNvbnN0IHByb3BUeXBlcyA9IHtcbiAgY2hpbGRyZW46IG9yKFtleHBsaWNpdE51bGwoKS5pc1JlcXVpcmVkLCBQcm9wVHlwZXMubm9kZS5pc1JlcXVpcmVkXSksXG59O1xuXG5jb25zdCBkZWZhdWx0UHJvcHMgPSB7XG4gIGNoaWxkcmVuOiB1bmRlZmluZWQsXG59O1xuXG5jb25zdCBXcmFwcGVyID0gKGludGVyc2VjdHMoJz49IDAuMTQnLCBSZWFjdC52ZXJzaW9uKVxuICAvLyBlc2xpbnQtZGlzYWJsZS1uZXh0LWxpbmUgcHJlZmVyLWFycm93LWNhbGxiYWNrXG4gID8gKCkgPT4gT2JqZWN0LmFzc2lnbihmdW5jdGlvbiBTaW1wbGVTRkNXcmFwcGVyKHsgY2hpbGRyZW4gfSkge1xuICAgIHJldHVybiBjaGlsZHJlbjtcbiAgfSwgeyBwcm9wVHlwZXMsIGRlZmF1bHRQcm9wcyB9KVxuICA6ICgpID0+IHtcbiAgICBjbGFzcyBTaW1wbGVDbGFzc1dyYXBwZXIgZXh0ZW5kcyBSZWFjdC5Db21wb25lbnQge1xuICAgICAgcmVuZGVyKCkge1xuICAgICAgICBjb25zdCB7IGNoaWxkcmVuIH0gPSB0aGlzLnByb3BzO1xuICAgICAgICByZXR1cm4gY2hpbGRyZW47XG4gICAgICB9XG4gICAgfVxuICAgIFNpbXBsZUNsYXNzV3JhcHBlci5wcm9wVHlwZXMgPSBwcm9wVHlwZXM7XG4gICAgU2ltcGxlQ2xhc3NXcmFwcGVyLmRlZmF1bHRQcm9wcyA9IGRlZmF1bHRQcm9wcztcbiAgICByZXR1cm4gU2ltcGxlQ2xhc3NXcmFwcGVyO1xuICB9XG4pKCk7XG5cbmV4cG9ydCBkZWZhdWx0IGZ1bmN0aW9uIHdyYXAoZWxlbWVudCkge1xuICByZXR1cm4gPFdyYXBwZXI+e2VsZW1lbnR9PC9XcmFwcGVyPjtcbn1cbiJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFDQTtBQUFtQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBRW5DLElBQU1BLFNBQVMsR0FBRztFQUNoQkMsUUFBUSxFQUFFLElBQUFDLG1CQUFFLEVBQUMsQ0FBQyxJQUFBQyw2QkFBWSxHQUFFLENBQUNDLFVBQVUsRUFBRUMscUJBQVMsQ0FBQ0MsSUFBSSxDQUFDRixVQUFVLENBQUM7QUFDckUsQ0FBQztBQUVELElBQU1HLFlBQVksR0FBRztFQUNuQk4sUUFBUSxFQUFFTztBQUNaLENBQUM7QUFFRCxJQUFNQyxPQUFPLEdBQUcsQ0FBQyxJQUFBQyxrQkFBVSxFQUFDLFNBQVMsRUFBRUMsaUJBQUssQ0FBQ0MsT0FBTztBQUNsRDtBQUFBLEVBQ0U7RUFBQSxPQUFNLHdCQUFjLFNBQVNDLGdCQUFnQixPQUFlO0lBQUEsSUFBWlosUUFBUSxRQUFSQSxRQUFRO0lBQ3hELE9BQU9BLFFBQVE7RUFDakIsQ0FBQyxFQUFFO0lBQUVELFNBQVMsRUFBVEEsU0FBUztJQUFFTyxZQUFZLEVBQVpBO0VBQWEsQ0FBQyxDQUFDO0FBQUEsSUFDN0IsWUFBTTtFQUFBLElBQ0FPLGtCQUFrQjtJQUFBO0lBQUE7SUFBQTtNQUFBO01BQUE7SUFBQTtJQUFBO01BQUE7TUFBQSxPQUN0QixrQkFBUztRQUNQLElBQVFiLFFBQVEsR0FBSyxJQUFJLENBQUNjLEtBQUssQ0FBdkJkLFFBQVE7UUFDaEIsT0FBT0EsUUFBUTtNQUNqQjtJQUFDO0lBQUE7RUFBQSxFQUo4QlUsaUJBQUssQ0FBQ0ssU0FBUztFQU1oREYsa0JBQWtCLENBQUNkLFNBQVMsR0FBR0EsU0FBUztFQUN4Q2Msa0JBQWtCLENBQUNQLFlBQVksR0FBR0EsWUFBWTtFQUM5QyxPQUFPTyxrQkFBa0I7QUFDM0IsQ0FBQyxHQUNBO0FBRVksU0FBU0csSUFBSSxDQUFDQyxPQUFPLEVBQUU7RUFDcEMsb0JBQU8sZ0NBQUMsT0FBTyxRQUFFQSxPQUFPLENBQVc7QUFDckM7QUFBQyJ9
//# sourceMappingURL=wrapWithSimpleWrapper.js.map